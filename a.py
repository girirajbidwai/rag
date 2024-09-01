import os
import logging
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import openai

# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "myindex"

# Global variables to keep track of state
chunks = []
model = None
index = None

def load_model():
    logging.info("Loading SentenceTransformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def init_pinecone():
    logging.info("Initializing Pinecone...")
    global index
    if index_name in pc.list_indexes().names():
        logging.info(f"Deleting existing Pinecone index: {index_name}")
        pc.delete_index(index_name)

    logging.info(f"Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    index = pc.Index(index_name)
    return index

def load_and_process_pdf(file_path):
    logging.info(f"Loading and processing PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_pinecone_index(chunks, model):
    logging.info("Creating Pinecone index with embeddings...")
    for i, chunk in enumerate(chunks):
        chunk_embedding = model.encode(chunk.page_content).tolist()
        index.upsert(vectors=[
            {
                'id': f'chunk_{i}',
                'values': chunk_embedding,
                'metadata': {'text': chunk.page_content}
            }
        ])
    logging.info("Pinecone index creation completed.")

# Initialize OpenAI client with your API key and endpoint
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://cloud.olakrutrim.com/v1"

def process_query(query: str, selected_model: str):
    logging.info(f"Processing query with model {selected_model}: {query}")
    
    global model, index
    
    if not model:
        model = load_model()
    
    if not index:
        index = init_pinecone()
    
    query_embedding = model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    top_chunks = [match['metadata']['text'] for match in result['matches']]
    context = "\n\n".join(top_chunks)
    
    prompt = f"Given the following context:\n\n{context}\n\nAnswer the following question:\n\n{query}"
    
    response = openai.ChatCompletion.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    logging.info("Query processed successfully.")
    return response.choices[0].message["content"]

def initialize(file_path):
    try:
        global model, chunks, index
        index = init_pinecone()
        model = load_model()
        chunks = load_and_process_pdf(file_path)
        create_pinecone_index(chunks, model)
        return "Initialization successful"
    except FileNotFoundError:
        error_msg = f"The specified PDF file '{file_path}' was not found."
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(error_msg)
        return error_msg

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            init_status = initialize(file_path)
            return render_template('index.html', status=init_status)
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        query_text = request.form['query']
        selected_model = request.form['model']
        logging.info(f"Received query: {query_text} and model: {selected_model}")
        answer = process_query(query_text, selected_model)
        return render_template('index.html', answer=answer, selected_model=selected_model)
    except KeyError as e:
        logging.error(f"KeyError: {str(e)}")
        return render_template('index.html', error="Missing required form fields.")

if __name__ == "__main__":
    app.run(debug=True)
