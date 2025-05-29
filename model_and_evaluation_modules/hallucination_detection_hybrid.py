# Includes: uncertainty_heuristic, nli_contradictions, ner_mismatches
# Sentence matching: multi-sentence aggregation using SBERT with entity-aware boosting

import sys
import json
import torch
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import re
from scipy.stats import entropy
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from transformers import logging
from tqdm import tqdm

logging.set_verbosity_error()

nlp = spacy.load("nb_core_news_sm")
punkt_param = PunktParameters()
norwegian_tokenizer = PunktSentenceTokenizer(punkt_param)

sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
nli_model_name = "alexandrainst/scandi-nli-large"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

# Defining thresholds
UNCERTAINTY_THRESHOLD_LOW = 0.6
UNCERTAINTY_THRESHOLD_HIGH = 0.8
NLI_THRESHOLD = 0.5
MAX_AGGREGATION = 3
MIN_GAIN = 0.05

def read_multiline_jsonl(file_path):
    entries, buffer = [], ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer += line
            try:
                entries.append(json.loads(buffer))
                buffer = ""
            except json.JSONDecodeError:
                continue
    return entries

def extract_entities(text):
    doc = nlp(text)
    return {ent.text.lower() for ent in doc.ents}

def compute_uncertainty_score(text):
    tokens = nli_tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        logits = nli_model(tokens).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    return entropy(probs.numpy(), base=2)

def split_sentences(text):
    return norwegian_tokenizer.tokenize(text)

def compute_sbert_similarity(original_text, rewritten_text, beam_width=3):
    original_sentences = split_sentences(original_text)
    rewritten_sentences = split_sentences(rewritten_text)

    matched_sentences = []

    for i, rewritten in enumerate(rewritten_sentences):
        rewrite_emb = sbert_model.encode(rewritten, convert_to_tensor=True)
        rewrite_ents = extract_entities(rewritten)

        similarities = util.pytorch_cos_sim(rewrite_emb, sbert_model.encode(original_sentences, convert_to_tensor=True))[0]
        top_indices = torch.topk(similarities, min(beam_width, len(original_sentences))).indices.tolist()

        beam = []
        for idx in top_indices:
            score = similarities[idx].item()
            sentence_text = original_sentences[idx]
            if rewrite_ents & extract_entities(sentence_text):
                score += 0.05
            beam.append({
                "indices": [idx],
                "score": score,
                "text": sentence_text
            })

        if not beam:
            matched_sentences.append({
                "rewritten_sentence": rewritten,
                "matched_original": "No Match Found",
                "similarity_score": 0.0
            })
            continue

        best_hypothesis = beam[0]

        for _ in range(len(original_sentences)):
            new_beam = []
            for hyp in beam:
                if len(hyp["indices"]) >= MAX_AGGREGATION:
                    continue
                used = set(hyp["indices"])
                for j in range(len(original_sentences)):
                    if j in used:
                        continue
                    new_indices = hyp["indices"] + [j]
                    combined_text = " ".join(original_sentences[k] for k in new_indices)
                    try:
                        combined_emb = sbert_model.encode(combined_text, convert_to_tensor=True)
                        score = util.pytorch_cos_sim(rewrite_emb, combined_emb)[0][0].item()
                        if rewrite_ents & extract_entities(original_sentences[j]):
                            score += 0.05
                        if score > hyp["score"] + MIN_GAIN:
                            new_beam.append({
                                "indices": new_indices,
                                "score": score,
                                "text": combined_text
                            })
                    except Exception as e:
                        print(f"Error during beam expansion: {e}")
            if not new_beam:
                break
            beam = sorted(new_beam, key=lambda x: x["score"], reverse=True)[:beam_width]
            best_hypothesis = beam[0]

        matched_sentences.append({
            "rewritten_sentence": rewritten,
            "matched_original": best_hypothesis["text"],
            "similarity_score": float(best_hypothesis["score"])
        })

    return matched_sentences

def detect_uncertainty_heuristic(matched):
    flags = []
    for pair in matched:
        r, o = pair["rewritten_sentence"], pair["matched_original"]
        re, oe = compute_uncertainty_score(r), compute_uncertainty_score(o)
        diff = float(abs(re - oe))
        threshold = UNCERTAINTY_THRESHOLD_HIGH if len(r.split()) <= 12 else UNCERTAINTY_THRESHOLD_LOW
        if diff > threshold:
            flags.append({"rewritten_sentence": r, "matched_original": o, "entropy_difference": diff})
    return flags

def detect_nli_contradictions(matched):
    flags = []
    for pair in matched:
        r, o = pair["rewritten_sentence"], pair["matched_original"]
        inputs = nli_tokenizer(o, r, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            score = float(probs[0][2].item())
        if score >= NLI_THRESHOLD:
            flags.append({"rewritten_sentence": r, "matched_original": o, "contradiction_score": score})
    return flags

def detect_ner_mismatches(matched):
    mismatches = []
    for pair in matched:
        ents_r = extract_entities(pair["rewritten_sentence"])
        ents_o = extract_entities(pair["matched_original"])
        if ents_r != ents_o:
            mismatches.append({"rewritten_sentence": pair["rewritten_sentence"], "matched_original": pair["matched_original"], "diff_entities": list(ents_r ^ ents_o)})
    return mismatches

def process_articles(input_path, output_path):
    data = read_multiline_jsonl(input_path)
    results = []
    for article in tqdm(data, desc="Processing articles"):
        matched = compute_sbert_similarity(article["original_text"], article["corrected_text"])
        result = {
            "unique_id": article["unique_id"],
            "title": article["title"],
            "matched_sentences": matched,
            "uncertainty_heuristic": detect_uncertainty_heuristic(matched),
            "nli_contradictions": detect_nli_contradictions(matched),
            "ner_mismatches": detect_ner_mismatches(matched)
        }
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise ValueError("Usage: python hallucination_detection_hybrid.py <input_file> <output_file>")

    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    
    process_articles(INPUT_FILE, OUTPUT_FILE)