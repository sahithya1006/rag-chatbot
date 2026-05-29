import streamlit as st
from groq import Groq

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# GROQ API
# -----------------------------

client = Groq(
    api_key="github_pat_1186RQ3310PSEazefAAjwP_Fq9Z5V5YmQeV0e84qvGDei4MgKtE7MldJoO"
)

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF RAG Chatbot")

st.write("Ask questions from your PDF")

# -----------------------------
# LOAD PDF
# -----------------------------

loader = PyPDFLoader("data/notes.pdf")

documents = loader.load()

# -----------------------------
# SPLIT PDF
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

embeddings = HuggingFaceEmbeddings()

# -----------------------------
# VECTOR DATABASE
# -----------------------------

db = FAISS.from_documents(
    docs,
    embeddings
)

# -----------------------------
# USER INPUT
# -----------------------------

prompt = st.text_input(
    "Ask a question from PDF"
)

# -----------------------------
# RESPONSE
# -----------------------------

if prompt:

    with st.spinner("Searching PDF..."):

        results = db.similarity_search(prompt)

        context = "\n".join(
            [doc.page_content for doc in results]
        )

        final_prompt = f"""
        Answer the question ONLY using this PDF content.

        PDF Content:
        {context}

        Question:
        {prompt}
        """

        chat_completion = client.chat.completions.create(

            messages=[
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ],

            model="llama-3.3-70b-versatile",
        )

        response = (
            chat_completion
            .choices[0]
            .message
            .content
        )

    st.success("Answer generated!")

    st.write(response)