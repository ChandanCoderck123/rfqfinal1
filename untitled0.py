# -*- coding: utf-8 -*-
import openai
import faiss
import numpy as np
import pandas as pd
import re
import json
from flask import Flask, request, jsonify

# Set your OpenAI API key
openai.api_key = "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"

# Set the embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text):
    text = text.replace("\n", " ").strip()
    try:
        # Use openai.Embedding.create from openai==0.28.0
        response = openai.Embedding.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def clean_text(text):
    # Remove non-ASCII characters and strip whitespace
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

# Load your CSV file here. Adjust path to point to your CSV on EC2.
csv_path = "SKU_list_of_23-24_Sheet1.csv"  # <-- Update this if needed
catalog_df = pd.read_csv(csv_path)

# Ensure columns exist. Adjust if your CSV differs.
catalog_df = catalog_df[['SKU', 'Brand', 'Description']].fillna("")

# Combine brand + description to form a single text for embedding
product_texts = catalog_df.apply(
    lambda x: f"{x['Brand']} {x['Description']}", axis=1
)
product_texts = product_texts.apply(clean_text)

embeddings_list = []
valid_indices = []

for idx, text in enumerate(product_texts):
    emb = get_embedding(text)
    if emb is not None:
        embeddings_list.append(emb)
        valid_indices.append(idx)

if not embeddings_list:
    raise ValueError("No embeddings generated. Check your API key or data format.")

# Build Faiss index
embeddings_array = np.vstack(embeddings_list)
faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
faiss_index.add(embeddings_array)

# Map each index back to the catalog entry
catalog_map = {
    i: catalog_df.iloc[valid_indices[i]].to_dict() for i in range(len(valid_indices))
}

app = Flask(__name__)

@app.route('/rfq', methods=['POST'])
def rfq_search():
    data = request.get_json()
    if not data or 'rfq' not in data:
        return jsonify({"error": "Invalid request. Provide 'rfq' field in JSON."}), 400

    rfq_input = data['rfq']
    rfq_lines = rfq_input.strip().split("\n")
    matched_products = []

    for line in rfq_lines:
        query = clean_text(line)
        query_embedding = get_embedding(query)
        if query_embedding is None:
            continue
        # Search top 5 closest embeddings
        _, indices = faiss_index.search(query_embedding.reshape(1, -1), 5)
        top_matches = []
        best_match = None

        for rank, match_idx in enumerate(indices[0]):
            if match_idx < 0:
                continue
            matched_row = catalog_map[match_idx]
            match_entry = {
                "rank": rank + 1,
                "product_id": matched_row["SKU"],
                "product_name": matched_row["Description"]
            }
            if rank == 0:
                best_match = match_entry
            top_matches.append(match_entry)

        matched_products.append({
            "original_string": line,
            "best_match": best_match,
            "top_5_matches": top_matches
        })

    return jsonify(matched_products), 200

if __name__ == "__main__":
    # Run on port 5000; host='0.0.0.0' so it's accessible externally
    app.run(host='0.0.0.0', port=5000, debug=True)
