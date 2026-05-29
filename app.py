import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")
st.write("Ask questions from your documents")

# -------------------------
# Load API Key
# -------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in environment variables")
    st.stop()

client = Groq(api_key=api_key)

# -------------------------
# Load file (your data)
# -------------------------
DATA_PATH = "data/info.txt"

if not os.path.exists(DATA_PATH):
    st.error("Data file not found in data/info.txt")
    st.stop()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# Split text
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# -------------------------
# Embeddings + FAISS
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embeddings)
retriever = db.as_retriever()

# -------------------------
# LLM function
# -------------------------
def get_answer(context, question):
    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------
# UI
# -------------------------
query = st.text_input("Ask something:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = " ".join([d.page_content for d in docs])

    answer = get_answer(context, query)

    st.subheader("Answer")
    st.write(answer)
