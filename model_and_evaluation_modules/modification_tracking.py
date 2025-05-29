import sys
import json
import pandas as pd
import nltk
import torch
import os
from rouge_score import rouge_scorer
import sacrebleu
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Manually set the NLTK data path
nltk.download('punkt')
nltk.data.path.append(os.path.expanduser("~/nltk_data"))

# Load Norwegian-specific tokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
norwegian_tokenizer = PunktSentenceTokenizer(punkt_param)

# Set Hugging Face cache directory to avoid storing models in Git
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

# Load tokenizer and model from cache
MODEL_TYPE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, cache_dir=CACHE_DIR)
model = AutoModel.from_pretrained(MODEL_TYPE, cache_dir=CACHE_DIR)

def read_multiline_jsonl(file_path):
    """
    Reads a JSONL file where entries can span multiple lines.
    Returns a list of parsed JSON objects.
    """
    entries = []
    buffer = ""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            buffer += line  # Add new line to buffer
            
            try:
                entry = json.loads(buffer)  # Try to parse the accumulated buffer
                entries.append(entry)  # If successful, add to list
                buffer = ""  # Reset buffer for next entry
            except json.JSONDecodeError:
                continue  # Keep accumulating lines if not a valid JSON object yet

    return entries


def compute_sbert_similarity(originals, rewrites):
    def get_embedding(text):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) # Mean pooling to get sentence embedding
    
    embeddings_original = torch.cat([get_embedding(text) for text in originals], dim=0)
    embeddings_rewrite = torch.cat([get_embedding(text) for text in rewrites], dim=0)

    # Compute cosine similarity between original and rewritten texts
    cosine_similarities = F.cosine_similarity(embeddings_original, embeddings_rewrite, dim=1)

    return cosine_similarities.tolist()

def compute_rouge_Lsum(original, rewrite):
    scorer = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(original, rewrite)
    return scores['rougeLsum'].fmeasure  # F1-score for content overlap

def compute_TER(original, rewrite):
    original_tokenized = norwegian_tokenizer.tokenize(original.lower())
    rewrite_tokenized = norwegian_tokenizer.tokenize(rewrite.lower())
    ter_score = sacrebleu.metrics.TER()
    score = ter_score.sentence_score(rewrite, [original]).score
    return score

def analyse_modifications(rewritten_file, output_file):
    data = read_multiline_jsonl(rewritten_file)

    original_texts = [entry["original_text"] for entry in data]
    corrected_rewritten_texts = [entry["corrected_text"] for entry in data]

    bert_scores = compute_sbert_similarity(original_texts, corrected_rewritten_texts)
    rouge_Lsum_scores = [compute_rouge_Lsum(orig, rewrite) for orig, rewrite in zip(original_texts, corrected_rewritten_texts)]
    TER_scores = [compute_TER(orig, rewrite) for orig, rewrite in zip(original_texts, corrected_rewritten_texts)]
    
    # Prepare results for saving
    results = []
    for i, entry in enumerate(data):
        results.append({
            "unique_id": entry["unique_id"],
            "title": entry["title"],
            "SBERT Cosine Similarity (Semantic Similarity)": bert_scores[i],
            "ROUGE-Lsum (Content Overlap)": rouge_Lsum_scores[i],
            "TER (Modification Distance)": TER_scores[i]
        })


     # Save results to a JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False, indent=4)
            f.write("\n")  # Ensure valid JSONL format

    print(f"Modification analysis completed. Results saved to {output_file}")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python modification_tracking.py <input_file> <output_file>")

    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]

    analyse_modifications(INPUT_FILE, OUTPUT_FILE)