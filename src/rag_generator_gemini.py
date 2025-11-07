import google.generativeai as genai
from rag_retriever import RAGRetriever
import os

# ✅ Load your Gemini API key

genai.configure(api_key=os.getenv("GEMINI_API_KEY") )

# ✅ Choose model
model = genai.GenerativeModel("models/gemini-2.5-flash")

class RAGPipeline:
    def __init__(self):
        print("✅ RAG Pipeline Ready (Retriever + Gemini)")
        self.retriever = RAGRetriever()

    # ---------------------------------------
    # Build a strict RAG prompt
    # ---------------------------------------
    def build_prompt(self, query, context_text):
        prompt = f"""
            You are a helpful assistant for GUVI.

            Answer the question STRICTLY using the following context. 
            Do NOT use any outside knowledge.
            If the answer is NOT present in the context, respond:
            "I couldn't find this information in GUVI resources."

            ---------------------
            CONTEXT:
            {context_text}
            ---------------------

            QUESTION: {query}

            ANSWER:
            """
        return prompt

    # ---------------------------------------
    # Generate final answer
    # ---------------------------------------
    def generate_answer(self, query, k=4):
        # Retrieve chunks
        context, docs = self.retriever.build_context(query, k=k)

        # Build prompt
        prompt = self.build_prompt(query, context)

        # Call Gemini
        response = model.generate_content(prompt)

        return response.text, docs


# -------------------------------------------------
# Test the full RAG pipeline
# -------------------------------------------------
if __name__ == "__main__":
    rag = RAGPipeline()

    user_q = input("\n🟢 Ask something: ")

    answer, sources = rag.generate_answer(user_q)

    print(answer)

    # print("\n======================")
    # print("📚 SOURCES USED:")
    # print("======================")
    # for src in sources:
    #     print(f"- {src['source']} | {src['title']}")
