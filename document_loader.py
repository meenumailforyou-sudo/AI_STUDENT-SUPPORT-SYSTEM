import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR


def load_documents():
    documents = []

    if not os.path.exists(DATA_DIR):
        print(f"Data folder not found: {DATA_DIR}")
        return documents

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)

            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                documents.extend(docs)

                print(f"Loaded: {filename}")

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")