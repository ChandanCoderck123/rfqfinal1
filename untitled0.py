import openai
import faiss
import numpy as np
import pandas as pd
import json
import re
import time

# Set OpenAI API key
openai.api_key = "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"

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# Function to generate embeddings
def get_embedding(text):
    text = text.replace("\n", " ").strip()
    try:
        response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# Function to clean text (remove special characters)
def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

# Load catalog master
catalog_df = pd.read_csv("./SKU list of 23-24 - Sheet1.csv")

# Select only required columns
catalog_df = catalog_df[['SKU', 'Brand', 'Description']].fillna("")

# Create product text for embeddings
product_texts = catalog_df.apply(lambda x: f"{x['Brand']} {x['Description']}", axis=1)
product_texts = product_texts.apply(clean_text)

# Generate embeddings for catalog products
embeddings_list = []
valid_indices = []

for idx, text in enumerate(product_texts):
    emb = get_embedding(text)
    if emb is not None:
        embeddings_list.append(emb)
        valid_indices.append(idx)

# Check if embeddings were generated
if not embeddings_list:
    raise ValueError("No embeddings generated. Check API key or data format.")

# Convert embeddings into FAISS index
embeddings_array = np.vstack(embeddings_list)
faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
faiss_index.add(embeddings_array)

# Mapping FAISS vector IDs back to catalog rows
catalog_map = {i: catalog_df.iloc[valid_indices[i]].to_dict() for i in range(len(valid_indices))}

# Example RFQ input
rfq_input = """
BINDER CLIP 25mm-INFINITY
CELLO TAPE 48mm
CALCULATOR-ORPAT
"""

# Process RFQ input
rfq_lines = rfq_input.strip().split("\n")
matched_products = []

for line in rfq_lines:
    query_embedding = get_embedding(clean_text(line))
    if query_embedding is None:
        continue

    _, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)  # Get top 5 matches
    top_matches = []

    for rank, match_idx in enumerate(indices[0]):
        if match_idx < 0:
            continue

        matched_row = catalog_map[match_idx]
        match_entry = {
            "rank": rank + 1,
            "product_id": matched_row["SKU"],
            "Product_name": matched_row["Description"]
        }

        if rank == 0:  # Highlight best match
            best_match = match_entry
        top_matches.append(match_entry)

    matched_products.append({
        "original_string": line,
        "best_match": best_match,
        "top_5_matches": top_matches
    })

# Final output in JSON format
output_json = json.dumps(matched_products, indent=2)
print(output_json)
