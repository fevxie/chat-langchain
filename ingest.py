"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os
import threading
import queue

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

# List all files in a directory recursively
def list_files_recursive(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def ingest_docs(file_path):
    """Get documents from web pages."""
    loader = UnstructuredMarkdownLoader(file_path)
    print(f'Starting Load file: {file_path}')
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    directory = '../docs'
    skip_ending = '.DS_Store'

    for file_path in list_files_recursive(directory):
        # Skip files with the specified ending
        if file_path.endswith(skip_ending):
            print(f'Skipping file: {file_path}')
            continue
        ingest_docs(file_path)
    print("All files processed.")
