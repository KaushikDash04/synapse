import os
import atexit
import io
from fastapi import FastAPI, File, UploadFile, Query
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

# Define the directory to store uploaded PDF files
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to extract text from PDF and create embeddings
def process_pdf(pdf_bytes: bytes) -> FAISS:
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

# Cleanup function to delete uploaded PDF files
def cleanup():
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Register the cleanup function to be called when the Python interpreter exits
atexit.register(cleanup)

# Endpoint to upload PDF file
@app.post("/upload_pdf/")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        pdf_bytes = await pdf.read()
        # Save the PDF file to the upload directory
        with open(os.path.join(UPLOAD_DIR, pdf.filename), "wb") as f:
            f.write(pdf_bytes)
        # Process the PDF file as needed
        knowledge_base = process_pdf(pdf_bytes)
        return {"message": "PDF uploaded successfully"}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to search for a query in the PDF files
@app.get("/search/{query}")
async def search(query: str = Query(...)):
    try:
        # Create an OpenAI language model
        openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        )
        # Create a conversational retrieval chain
        chain = ConversationalRetrievalChain(
            vector_store=knowledge_base,
            language_model=openai,
            callbacks=[get_openai_callback(openai)]
        )
        # Search for the query in the PDF files
        response = chain.search(query)
        return response
    except Exception as e:
        return {"error": str(e)}