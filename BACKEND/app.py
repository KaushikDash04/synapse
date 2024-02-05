import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import IPython.display as ipd

import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

import google.generativeai as genai
import PIL.Image

from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

api_key = "AIzaSyBPswRnrMGDkpzrBezeQZTUY-t0ZZGVtbI"
genai.configure(api_key=api_key)

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

def generate_text_vision(question):
    genai.configure(api_key=api_key)

    img = PIL.Image.open('photos\hello.jpg')

    print("User Question: ", question)

    model = genai.GenerativeModel(model_name="gemini-pro-vision")
    response = model.generate_content([question, img])
    print("AI Answer: ")
    print(response.text)
    return response.text

def generate_text_gemini(question):
    print("User Question: ", question)
    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content([question])
    print("AI Answer: ")
    print(response.text)
    return response.text

def generate_text_pdf(question):
    print("User Question: ", question)
    model = genai.GenerativeModel(model_name="pdf-gpt")
    response = model.generate_content([question])
    print("AI Answer: ")
    print(response.text)
    return response.text

# Endpoint to receive GET requests
# @app.get("/generate_text_gemini/")
# async def get_generated_text(question: str, description: str):
#     processed_data = generate_text_gemini(question)
#     return {"User Question" : question,
#             "Description" :  description,
#             "AI Response": processed_data}

# @app.get("/generate_text_vision/")
# async def get_generated_text(question: str):
#     processed_data = generate_text_vision(question)
#     return {"User Question" : question, 
#             "AI Response": processed_data}

@app.get("/generate_text_gemini/")
async def get_generated_text(model_name: str, question: str):
    if model_name == "gemini-pro":
        processed_data = generate_text_gemini(question)
    elif model_name == "gemini-vision":
        processed_data = generate_text_vision(question)
    elif model_name == "pdf-gpt":
        processed_data = generate_text_pdf(question)
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")
        
    return {"Question" : question,
            "Model Name" : model_name,
            "AI Response": processed_data}
