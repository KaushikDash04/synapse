# google gen ai imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# PIL imports
import PIL.Image
import uuid

#fastapi imports
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from starlette.middleware.cors import CORSMiddleware

#PDF GPT imports
import os
import atexit
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# environment variables
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

genai.configure()


# Generating text for the image input (Gemini Vision model)
def generate_text_vision(question):
    genai.configure()

    img = PIL.Image.open(image_path)

    print("User Question: ", question)

    model = genai.GenerativeModel(model_name="gemini-pro-vision")
    response = model.generate_content([question, img])
    print("AI Answer: ")
    print(response.text)
    return response.text

# Generating text for the conversation input (Gemini pro text model)
def generate_text_gemini(question):
    print("User Question: ", question)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([question])
    print("AI Answer: ")
    print(response.text)
    return response.text

# Generating text for the PDF input (PDF GPT model)
def generate_text_pdf(question, vector_store):
    try:
        docs = vector_store.similarity_search(question)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": question}, return_only_outputs=True)
        return(response["output_text"])
    except Exception as e:
        return {"error": str(e)}


# Function to load the conversational chain
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just, "summarize the document and write everything", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to process PDF file and create embeddings
def process_pdf(pdf_bytes):
    try:
        text = ""
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        text_chunks = text_splitter.split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        return vector_store
    except Exception as e:
        raise Exception("Error processing PDF")  # Raise an exception instead of returning an error dictionary


# Create a directory to store uploaded files    
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Function to process uploaded image
def image_path_generate(image_bytes: bytes) -> str:
    # Generate unique filename
    filename = f"image_{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Save image file
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # Return filepath
    return filepath


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
        global vector_space 
        vector_space = process_pdf(pdf_bytes)
        print("PDF processed successfully")
        return {"message": "PDF uploaded successfully"}
    except Exception as e:
        return {"error": str(e)}
    
# Endpoint to upload Image file
@app.post("/upload_image/")
async def upload_image(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        global image_path 
        image_path = image_path_generate(image_bytes)
        return {"message": "Image uploaded successfully", "image_path": image_path}
    except Exception as e:
        return {"error": str(e)}

@app.get("/generate_text_gemini/")
async def get_generated_text(model_name: str, question: str):
    if model_name == "gemini-pro":
        processed_data = generate_text_gemini(question)
    elif model_name == "gemini-vision":
        processed_data = generate_text_vision(question)
    elif model_name == "pdf-gpt":
        processed_data = generate_text_pdf(question, vector_space)
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")
        
    return {"Question" : question,
            "Model Name" : model_name,
            "AI Response": processed_data}





# import torch
# import soundfile as sf
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import IPython.display as ipd

# import sounddevice as sd
# import soundfile as sf
# from pydub import AudioSegment

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI


# Function to extract text from PDF and create embeddings
# def process_pdf(pdf_bytes: bytes) -> FAISS:
#     pdf_stream = io.BytesIO(pdf_bytes)
#     pdf_reader = PdfReader(pdf_stream)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     embeddings = OpenAIEmbeddings()
#     knowledge_base = FAISS.from_texts(chunks, embeddings)
#     return knowledge_base



# Generating text for the PDF input (PDF GPT model)
# def generate_text_pdf(question):
#     try:
#         # Process uploaded PDF files
#         llm = ChatOpenAI()
#         memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#         # openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
#         chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             retriever=vector_space.as_retriever(),
#             memory = memory
#         )
#         response = chain({'question': question})
#         answer = response['chat_history'][-1].content
#         return answer
#     except Exception as e:
#         return {"error": str(e)}


# def load_voice_model():
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # model_id = "openai/whisper-large-v3"

    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     model_id, torch_dtype=torch_dtype, use_safetensors=True
    # )
    # model.to(device)

    # processor = AutoProcessor.from_pretrained(model_id)

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     max_new_tokens=128,
    #     chunk_length_s=30,
    #     batch_size=16,
    #     return_timestamps=True,
    #     torch_dtype=torch_dtype,
    #     device=device,
    # )

# def generate_text():
#     result = pipe("output.mp3")
#     print(result["text"])
#     return result["text"]

# prompt = generate_text()
