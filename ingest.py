import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(override=True)

def main():
    print("Loading PDFs...")
    docs = []
    for f in glob("data/*.pdf"):
        print(f" -> {f}")
        docs.extend(PyPDFLoader(f).load())
    
    if not docs:
        print("No PDFs found in data/")
        return

    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    Chroma.from_documents(
        documents=chunks,
        embedding=MistralAIEmbeddings(),
        persist_directory="chroma_db"
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
