from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone  # Updated import
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import promptTemplate  # Ensure this is defined in src.prompt
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the .env file")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)  # Create an instance of the Pinecone class

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Define the index name for Pinecone
index_name = "medical-chat"

# Initialize Pinecone index
index = pc.Index(index_name)

# Initialize the Langchain Pinecone vector store without the host argument
docsearch = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Define the prompt for the LLM
PROMPT = PromptTemplate(template=promptTemplate, input_variables=["context", "question"])

# Configuration for the chain
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the LLM
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Ensure this file exists
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


