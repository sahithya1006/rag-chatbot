import streamlit as st
import os
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.write("Ask questions from your PDF document")

# -------------------------
# API KEY (Streamlit Secrets)
# -------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets")
    st.stop()

client = Groq(api_key=api_key)

# -------------------------
# LOAD PDF
# -------------------------
DATA_PATH = "data/nodes.pdf"

if not os.path.exists(DATA_PATH):
    st.error("File not found: data/nodes.pdf")
    st.stop()

reader = PdfReader(DATA_PATH)

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

text = text.strip()

if len(text) == 0:
    st.error("PDF has no readable text. Use a text-based PDF (not scanned image PDF).")
    st.stop()

# -------------------------
# SPLIT TEXT
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

if len(chunks) == 0:
    st.error("No text chunks generated from PDF")
    st.stop()

# -------------------------
# EMBEDDINGS + FAISS
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_texts(chunks, embeddings)
retriever = vector_db.as_retriever()

# -------------------------
# GROQ LLM (FIXED + STABLE)
# -------------------------
def get_answer(context, question):

    if not question or question.strip() == "":
        return "Please enter a valid question."

    if not context or len(context.strip()) == 0:
        return "No relevant context found in the document."

    context = context[:2000]  # safety limit

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers ONLY from the given context."
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}
"""
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"API Error: {str(e)}"

# -------------------------
# CHAT UI
# -------------------------
query = st.text_input("Ask your question:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = " ".join([d.page_content for d in docs])

    answer = get_answer(context, query)

    st.subheader("Answer")
    st.write(answer)