import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os   

st.set_page_config(page_title="AskAI", layout="centered")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}
.card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)



def load_docs():
    docs = []
    folder = "docs"
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        # only open/s files, not folders
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    
    return docs
 
st.markdown('<div class="title">🤖 AskAI — Your Smart Knowledge Assistant</div>', unsafe_allow_html=True)

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
        background: linear-gradient(135deg, #d4fc79, #96e6a1);
        padding:18px;
        border-radius:12px;
        color:#1b5e20;
        font-size:16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        ">
        {answer}
        </div>
        """, unsafe_allow_html=True)
