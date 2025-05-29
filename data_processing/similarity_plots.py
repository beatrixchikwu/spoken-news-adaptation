import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

similarity_file = '/Users/Beatrix/Documents/UiB/MS_Informasjonsvitenskap/Masteroppgave/spoken-news-adaption/data_processing/SBERT_similarity_data_multilingual.json'

# Load similarity data
with open(similarity_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract values
similarities_matrix = np.array(data["similarities_matrix"])  # Full similarity matrix
top_indices_per_example = data["top_indices_per_example"]  # Indices of selected articles
top_ids = data["top_ids"]  # Unique IDs of selected articles
fine_tuning_ids = data["fine_tuning_ids"]  # Unique IDs of fine-tuning examples

print(f"Heatmap Matrix Shape: {similarities_matrix.shape}")
print(f"Selected candidate articles: {len(top_ids)}")
print(f"Reference articles: {len(fine_tuning_ids)}")

# Reshape heatmap data 
heatmap_data = similarities_matrix[top_indices_per_example]  

# Format labels: "candidate/reference + number"
formatted_article_labels = [f"Candidate {i+1}" for i in range(len(top_ids))]
formatted_fine_tuning_labels = [f"Reference {i+1}" for i in range(len(fine_tuning_ids))]

# Plot multiple histograms (one per fine-tuning example)
fig, axes = plt.subplots(5, 2, figsize=(5, 10), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten 2D grid into a list
for i in range(10):  # Loop over 10 fine-tuning examples
    ax = axes[i]
    ax.hist(similarities_matrix[:, i], bins=20, alpha=0.75, color='skyblue', edgecolor='darkslategrey', linewidth=0.5)
    ax.set_title(f"Reference article {i+1}", fontsize=12)
    ax.grid(True, linewidth=0.3) 
    ax.set_xlabel("Similarity Score", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)

plt.tight_layout()
plt.show()

# Plot heatmap
plt.figure(figsize=(6, 12))
ax = sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.7}, annot_kws={"fontsize":10})

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)

# Update x/y axis labels
ax.set_yticks(np.arange(len(top_ids)) + 0.5)
ax.set_yticklabels(formatted_article_labels, rotation=0, fontsize=10, ha="right")
ax.set_xticks(np.arange(len(fine_tuning_ids)) + 0.5)
ax.set_xticklabels(formatted_fine_tuning_labels, rotation=45, fontsize=10)

plt.ylabel("Top 39 Retrieved Candidate Articles", fontsize=12)
plt.xlabel("Reference Articles", fontsize=12)
plt.title("Heatmap of Cosine Similarity Scores Between Reference Articles and Selected Candidate Articles", fontsize=15, pad=20)
plt.show()