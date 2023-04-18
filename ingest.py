"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores.faiss import FAISS


# List all files in a directory recursively
def list_files_recursive(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def ingest_docs(file_path):
    """Get documents from web pages."""
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    print(f'Starting Load file: {file_path}')
    data = loader.load()

    markdown_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    documents = markdown_splitter.split_documents(data)
    # documents = markdown_splitter.create_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    directory = '../docs'
    skip_ending = '.DS_Store'
    #
    # os.environ["http_proxy"] = "http://127.0.0.1:1087"
    # os.environ["https_proxy"] = "http://127.0.0.1:1087"

    for file_path in list_files_recursive(directory):
        # Skip files with the specified ending
        if file_path.endswith(skip_ending):
            print(f'Skipping file: {file_path}')
            continue
        ingest_docs(file_path)
    print("All files processed.")
