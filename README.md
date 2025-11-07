# 🤖 GUVI Knowledge Retrieval Chatbot (RAG + Gemini)

An intelligent RAG-based chatbot built for the GUVI EdTech platform.  
It retrieves accurate answers from GUVI Blogs + FAQs using:

✅ FAISS Vector Store  
✅ Sentence-Transformers Embeddings  
✅ Google Gemini (models/gemini-2.5-flash)  
✅ Streamlit UI  
✅ Chunking + Retrieval-Augmented Generation (RAG)

---

## 🚀 Features

- RAG pipeline (Retriever + Generator)
- Fast semantic search using FAISS
- Google Gemini for natural, grounded answers
- Blog + FAQ dataset combined
- Automatic chunking + embedding generation
- Evaluation metrics:
  - Precision@k
  - Recall@k
  - BLEU / ROUGE
  - Latency
- Streamlit Chat UI

---

## 📁 Project Structure

guvi-rag-chatbot/
│
├── src/
│ ├── rag_retriever.py
│ ├── rag_generator_gemini.py
│ ├── chunk_data.py
│ ├── embed_data.py
│ ├── evaluate_rag.py
│ └── streamlit_app.py
│
├── vector_store/
│ ├── faiss.index
│ └── metadata.pkl
│
├── data/
│ └── combined_dataset.csv
│
├── requirements.txt
├── .gitignore
└── README.md


---

## ✅ Installation

### 1. Clone the repository
git clone https://github.com/suryak1904/guvi-rag-chatbot.git
cd guvi-rag-chatbot



### 2. Install dependencies
pip install -r requirements.txt



### 3. Add your API key  
Create a `.env` file:

GEMINI_API_KEY=your_api_key_here


✅ This file is not tracked by GitHub

---

## ✅ Running the Streamlit App

streamlit run src/streamlit_app.py


---

## ✅ Evaluation Metrics

Run retrieval + generation evaluation:

python src/evaluate_rag.py


---

## 📦 Technologies Used

- Python  
- Streamlit  
- FAISS  
- Sentence-Transformers  
- Google Gemini API  
- Pandas / NumPy  

---

## ✨ Author

**Surya K**  
AI / ML / Data Engineering  

---


