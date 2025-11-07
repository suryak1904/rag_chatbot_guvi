import streamlit as st
from rag_generator_gemini import RAGPipeline

# ----------------------------------
# Initialize RAG Pipeline once
# ----------------------------------
@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

rag = load_rag_pipeline()

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="GUVI Knowledge Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 GUVI RAG Chatbot")
st.caption("Instant answers based on GUVI Blogs & FAQs (RAG + Gemini)")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# User input
prompt = st.chat_input("Ask something about GUVI…")

if prompt:
    # Show user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # RAG Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, _ = rag.generate_answer(prompt)   # IGNORE SOURCES
            st.write(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
