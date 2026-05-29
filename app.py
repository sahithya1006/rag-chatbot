import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os

st.title("RAG Chatbot")

# Load data
with open("data/info.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embeddings)
retriever = db.as_retriever()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(question, context):
    prompt = f"""
You are a helpful assistant.
Use the context below to answer.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

query = st.text_input("Ask something from your data")

if query:
    docs = retriever.get_relevant_documents(query)
    context = " ".join([d.page_content for d in docs])

    answer = ask_llm(query, context)
    st.write(answer)