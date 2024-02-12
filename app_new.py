import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import huggingface_hub

from fastapi import FastAPI
from typing import List

app = FastAPI()

def get_pdf_text(pdf_name: str) -> str:
    with open(pdf_name, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: List[str]):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def generate_response(user_question: str, pdf_name: str) -> str:
    load_dotenv()
    raw_texts = get_pdf_text(pdf_name)
    text_chunks = get_text_chunks(raw_texts)
    vector_store = get_vector_store(text_chunks)
    llm = huggingface_hub.HuggingFaceHub(repo_id="grammarly/coedit-large", model_kwargs={"temperature": 0.5, "max_length": 512})
    response = llm(prompt=user_question, retriever=vector_store.as_retriever())
    return response['responses'][0]['text']



response = generate_response("What is ChatGPT?", "sample.pdf")
print(response)

# Endpoint to receive GET requests
@app.get("/generate_response/")
async def get_generated_response(question: str, pdf_file: str):
    response = generate_response(question, pdf_file)
    return {"AI Response": response}
