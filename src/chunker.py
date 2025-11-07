import pandas as pd
import re
import json
import os

RAW_CSV = "data/raw/guvi_combined_data.csv"
OUT_CSV = "data/processed/chunks.csv"

os.makedirs("data/processed", exist_ok=True)


# CLEAN TEXT

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)   
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

# CHUNK GENERATOR
def chunk_text(text, max_chars=500, overlap=100):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start += max_chars - overlap  

    return chunks

# MAIN CHUNKING PROCESS
def create_chunks():
    print("📥 Loading dataset:", RAW_CSV)
    df = pd.read_csv(RAW_CSV)

    chunk_records = []

    for idx, row in df.iterrows():
        combined_text = clean_text(str(row["content"]))

        chunks = chunk_text(combined_text)

        for i, ch in enumerate(chunks):
            chunk_records.append({
                "source": row["category"],
                "title": row["title"],
                "url": row["url"],
                "chunk_id": f"{row['category']}_{idx}_chunk_{i}",
                "chunk": ch
            })

    pd.DataFrame(chunk_records).to_csv(OUT_CSV, index=False)

    print(f"✅ Created {len(chunk_records)} chunks")
    print("✅ Saved to data/processed/")


if __name__ == "__main__":
    create_chunks()

