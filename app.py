import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

import os   

def load_docs():
    docs = []
    folder = "docs"
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        # ✅ IMPORTANT: only open files, not folders
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    
    return docs

##st.title("🛠️ DevOps AI Assistant")
st.title("AskAI: Multi-Domain Knowledge Assistant")

# Step 1: Split text
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)

# load files
documents = load_docs()
##st.write("Loaded docs:", documents)

# split all docs
# split all docs properly
texts = []

for doc in documents:
    chunks = text_splitter.split_text(doc)
    texts.extend(chunks)
##st.write("Texts:", texts)

##st.write("Texts:", texts)

# Step 2: Free embeddings (no API)
@st.cache_resource
def load_vector_db(texts):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)
    return db

db = load_vector_db(texts)

# UI
query = st.text_input("Ask a question:")


if query:
    docs = db.similarity_search(query, k=1)

    if not docs:
        st.warning("No relevant answer found.")
    else:
        context = docs[0].page_content
        answer = context.strip()

        st.subheader("🤖 Answer")

        st.markdown(f"""
        <div style="
        background-color:#e8f5e9;
        padding:15px;
        border-radius:10px;
        color:#1b5e20;
        font-size:16px;
        ">
        {answer}
        </div>
        """, unsafe_allow_html=True)