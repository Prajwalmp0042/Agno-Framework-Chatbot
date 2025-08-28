import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables (optional, if you need for later)
load_dotenv()

def ingest_repo(repo_path="agno"):
    print("🔍 Loading Markdown & Python Files...")

    # Load markdown and Python files
    md_loader = DirectoryLoader(repo_path, glob="**/*.md", recursive=True)
    py_loader = DirectoryLoader(repo_path, glob="**/*.py", recursive=True)

    docs = md_loader.load() + py_loader.load()
    print(f"📄 Total documents loaded: {len(docs)}")

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪 Documents split into {len(chunks)} chunks.")

    # Embeddings using HuggingFace (free, no API key required)
    print("🔎 Creating embeddings with HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the vectorstore
    print("💾 Saving vectorstore to 'agno_llama_index/'...")
    vectorstore.save_local("agno_llama_index")
    print("✅ Done!")

if __name__ == "__main__":
    ingest_repo()
