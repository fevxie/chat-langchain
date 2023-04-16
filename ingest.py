"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os
import threading
import queue
import chardet

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


# Worker function to process files
# def process_file(file_path):
#     with open(file_path, 'r') as f:
#         content = f.read()
#         # Perform any processing or operations on the content here
#         print(f"Processing {file_path}: {content[:100]}")

# Worker thread function
def worker_thread(file_queue):
    while True:
        file_path = file_queue.get()
        if file_path is None:
            break
        ingest_docs(file_path)
        file_queue.task_done()

# List all files in a directory recursively
def list_files_recursive(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def ingest_docs(file_path):
    """Get documents from web pages."""
    loader = UnstructuredMarkdownLoader(file_path)
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

    # Set up the file queue and worker threads
    num_threads = 1
    file_queue = queue.Queue()
    threads = []
    skip_ending = '.DS_Store'

    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread, args=(file_queue,))
        t.start()
        threads.append(t)

    # Add files to the queue
    for file_path in list_files_recursive(directory):
        # Skip files with the specified ending
        if file_path.endswith(skip_ending):
            print(f'Skipping file: {file_path}')
            continue
        file_queue.put(file_path)

    # Wait for all files to be processed
    file_queue.join()

    # Add sentinel values to signal the worker threads to exit
    for _ in range(num_threads):
        file_queue.put(None)

    # Wait for all worker threads to finish
    for t in threads:
        t.join()

    print("All files processed.")
    # ingest_docs()
