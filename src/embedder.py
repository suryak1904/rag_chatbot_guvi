import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

CHUNK_FILE = "data/processed/chunks.csv"
INDEX_FILE = "data/processed/faiss_index.bin"
META_FILE = "data/processed/metadata.json"

os.makedirs("data/processed", exist_ok=True)

# Load embedding model
def load_model():
    print(" Loading embedding model: all-MiniLM-L6-v2 ...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings
def create_embeddings(model, texts):
    print(f" Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(" FAISS index created!")
    return index


# Save FAISS index + metadata
def save_index(index, metadata):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(" Saved FAISS index and metadata")


# Main
def main():
    df = pd.read_csv(CHUNK_FILE)
    print(f" Loaded {len(df)} chunks")
    model = load_model()
    texts = df["chunk"].tolist()
    embeddings = create_embeddings(model, texts)
    index = build_faiss_index(embeddings)
    metadata = df.to_dict(orient="records")
    save_index(index, metadata)

    print("\n Embedding + FAISS pipeline complete!")


if __name__ == "__main__":
    main()

