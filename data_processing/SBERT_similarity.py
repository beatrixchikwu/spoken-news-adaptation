import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Model loaded successfully!")

preprocessed_file = '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/articles_preprocessed.jsonl'
core_dataset = '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/core_dataset.jsonl'
output_file = '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/40_most_similar_articles.jsonl'
similarity_output = '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/master-project-code/data_processing/SBERT_similarity_data.json'

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Function to read multi-line JSONL (for few-shot context set)
def read_multiline_jsonl(file_path):
    entries = []
    buffer = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  
            buffer += line  
            try:
                entry = json.loads(buffer)
                entries.append(entry)
                buffer = ""  
            except json.JSONDecodeError:
                continue
    if buffer.strip():
        print(f"Warning: Incomplete JSON detected at the end of file:\n{buffer}")
    return entries

# Start timing
start_time = time.time()

# Load articles
preprocessed_articles = load_jsonl(preprocessed_file)
print(f"Loaded {len(preprocessed_articles)} articles.")

core_dataset_articles = read_multiline_jsonl(core_dataset)
print(f"Loaded {len(core_dataset_articles)} fine-tuning articles.")

# Extract text content
preprocessed_texts = [article["original_text"] for article in preprocessed_articles]
fine_tuning_texts = [article["original_text"] for article in core_dataset_articles]

# Generate embeddings
preprocessed_embeddings = model.encode(preprocessed_texts, batch_size=32, show_progress_bar=True)
print(f"Generated embeddings for {len(preprocessed_texts)} articles.")

fine_tuning_embeddings = model.encode(fine_tuning_texts, batch_size=32, show_progress_bar=True)
print(f"Generated embeddings for {len(fine_tuning_texts)} fine-tuning articles.")

# Compute similarity matrix: (1598 articles x 10 fine-tuning examples)
similarities_matrix = cosine_similarity(preprocessed_embeddings, fine_tuning_embeddings)

# Select top 4 articles per fine-tuning example 
top_indices_per_example = []
for i in range(10):  
    top_indices = np.argsort(similarities_matrix[:, i])[-4:][::-1]  
    top_indices_per_example.extend(top_indices)

# Remove duplicates out of the 40 articles
top_indices_per_example = list(dict.fromkeys(top_indices_per_example))[:40]

# Get selected articles and IDs
top_articles = [preprocessed_articles[i] for i in top_indices_per_example]
top_ids = [preprocessed_articles[i]["unique_id"] for i in top_indices_per_example]
core_dataset_id = [article["unique_id"] for article in core_dataset_articles]

with open(output_file, 'w', encoding='utf-8') as f:
    for article in top_articles:
        f.write(json.dumps(article, ensure_ascii=False) + '\n')


# Convert NumPy types to standard Python types before saving
similarity_data = {
    "similarities_matrix": similarities_matrix.tolist(),  # Convert full matrix to list
    "top_indices_per_example": [int(i) for i in top_indices_per_example],  # Convert int64 to int
    "top_ids": top_ids,
    "fine_tuning_ids": core_dataset_id
}

with open(similarity_output, 'w', encoding='utf-8') as f:
    json.dump(similarity_data, f, ensure_ascii=False, indent=4)

print(f"Selected 40 articles and saved to '{output_file}'")
print(f"Similarity data saved to '{similarity_output}'")
print(f"Total execution time: {time.time() - start_time:.2f} seconds.")