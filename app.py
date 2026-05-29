import streamlit as st
import os
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# -------------------------
# UI setup
# -------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.write("Ask questions from your PDF document")

# -------------------------
# API Key
# -------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY in Streamlit secrets")
    st.stop()

client = Groq(api_key=api_key)

# -------------------------
# Load PDF safely
# -------------------------
DATA_PATH = "data/nodes.pdf"

if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

reader = PdfReader(DATA_PATH)

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

text = text.strip()

# IMPORTANT SAFETY CHECK
if len(text) == 0:
    st.error("PDF has no readable text. It may be a scanned/image PDF.")
    st.stop()

# -------------------------
# Split into chunks
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# IMPORTANT SAFETY CHECK
if len(chunks) == 0:
    st.error("No text chunks created from PDF")
    st.stop()

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_texts(chunks, embeddings)
retriever = vector_db.as_retriever()

# -------------------------
# LLM function
# -------------------------
def get_answer(context, question):
    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# -------------------------
# Chat UI
# -------------------------
query = st.text_input("Ask your question:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = " ".join([d.page_content for d in docs])

    answer = get_answer(context, query)

    st.subheader("Answer")
    st.write(answer)