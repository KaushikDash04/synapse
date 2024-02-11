import os
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, File, UploadFile, Query
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Define directory to store uploaded PDF files
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to extract text from PDF and create embeddings
def process_pdf(pdf_bytes: bytes) -> FAISS:
    pdf_stream = io.BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(
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
import atexit
atexit.register(cleanup)

# @app.on_event("startup")
# async def load_kb():

# Endpoint to upload PDF file
@app.post("/upload_pdf/")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        pdf_bytes = await pdf.read()
        # Save the PDF file to the upload directory
        with open(os.path.join(UPLOAD_DIR, pdf.filename), "wb") as f:
            f.write(pdf_bytes)
        # Process the PDF file as needed
        global vector_space 
        vector_space = process_pdf(pdf_bytes)
        return {"message": "PDF uploaded successfully"}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to search for a query in the PDF files
@app.get("/search/")
async def search(query: str = Query(...)):
    try:
        # Process uploaded PDF files
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        # openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_space.as_retriever(),
            memory = memory
        )
        response = chain({'question': query})
        answer = response['chat_history'][-1].content
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}
