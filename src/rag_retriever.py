import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/processed/faiss_index.bin"
META_FILE = "data/processed/metadata.json"


class RAGRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("🔍 Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("📦 Loading FAISS index...")
        self.index = faiss.read_index(INDEX_FILE)

        print("📄 Loading metadata...")
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # -----------------------------------
    # Embed user query
    # -----------------------------------
    def embed_query(self, query):
        return np.array(self.model.encode([query])).astype("float32")

    # -----------------------------------
    # Search FAISS
    # -----------------------------------
    def search(self, query, k=5):
        query_vec = self.embed_query(query)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results

    # -----------------------------------
    # Build context for LLM
    # -----------------------------------
    def build_context(self, query, k=5):
        docs = self.search(query, k)

        context = ""
        for d in docs:
            context += f"[{d['source']}] {d['title']}:\n{d['chunk']}\n\n"

        return context, docs


# -----------------------------------
# Testing the retriever
# -----------------------------------
if __name__ == "__main__":
    retriever = RAGRetriever()

    user_q = input("\nAsk something: ")

    context, docs = retriever.build_context(user_q)

    print("\n🔍 Retrieved Context:\n")
    print(context)
