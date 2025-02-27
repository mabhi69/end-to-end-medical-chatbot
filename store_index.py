from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Ensure the extracted data from PDF and text chunking is correct
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Downloading HuggingFace embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "medical-chat"

# Check if the index exists and create if it doesn't
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,  # Make sure this matches the dimension of your embeddings
        metric="cosine",  # You can change this metric if needed (e.g., 'euclidean')
    )

# Connect to the existing index
index = pc.Index(index_name)

# Ensure text_chunks is a list of raw strings
if not isinstance(text_chunks, list) or not all(isinstance(t, str) for t in text_chunks):
    if hasattr(text_chunks[0], 'page_content'):
        text_chunks = [t.page_content for t in text_chunks]
    else:
        raise ValueError("text_chunks must be a list of strings or have 'page_content' attribute.")

# Convert text chunks into embeddings and store them in Pinecone using LangchainPinecone
docsearch = LangchainPinecone.from_texts(
    texts=text_chunks,  # The text chunks as raw strings
    embedding=embeddings,
    index_name=index_name
)

print("Documents indexed successfully!")
