import os
import time
import logging
import speech_recognition as sr
import base64
import threading
import tempfile
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import pandas as pd
import gradio as gr
from groq import Groq
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"googlekey.json"  
translate_client = translate.Client()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"googlekey2.json"  
text_to_speech_client = texttospeech.TextToSpeechClient()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'sv2_pt_30d8bc22e01048e6ac9036fbcff466d2_ffa1e3c34d'

pc = Pinecone(api_key="9f8aa1d2-88ea-41a8-a731-bde767f0bfff")
pinecone_index_name = "flipkart-products"

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

GOOGLE_API_KEY = 'AIzaSyD28J8FyR2NI65oGvI9DHJmjlWUxEpDZUI'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

if 'flipkart-products' not in pc.list_indexes().names():
    pc.create_index(
        name='flipkart-products',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pc.Index('flipkart-products')

df = pd.read_csv("TVSCredit_FAQs - Combined (1).csv")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
df['id'] = df['id'].astype('object')
for i in range(len(df)):
    string = df.iloc[i]['Questions']
    embeddings = genai.embed_content(
        model="models/text-embedding-004",
        content=string
    )
    index.upsert(
        vectors=[
            {"id": "prod" + str(i), "values": embeddings['embedding']}
        ]
    )
    df.at[i, 'id'] = "prod" + str(i)


interrupt_flag = False
playback_thread = None
chat_history = []

def stop_playback():
    global interrupt_flag, playback_thread
    if playback_thread and playback_thread.is_alive():
        interrupt_flag = True
        playback_thread.join()

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file)
    return transcription.text


def transcribe_audio_step(audio_file):
    if audio_file is None:
        return "No audio file received."

    
    audio_file_path = audio_file  
    user_input = transcribe_audio(audio_file_path)

    return user_input
def translate_to_english(text):
    detection = translate_client.detect_language(text)
    detected_language = detection['language']

    if detected_language == "hi":  
        logging.info("Detected Hindi, translating to English...")
        translation = translate_client.translate(text, target_language="en")
        return translation['translatedText'], detected_language
    
    
    logging.info("Detected English, no translation needed.")
    return text, detected_language


def translate_to_original_language(text, target_language):
    translation = translate_client.translate(text, target_language=target_language)
    return translation['translatedText']

def query_pinecone(user_query):
    embeddings = genai.embed_content(model="models/text-embedding-004", content=user_query)
    
    query_results = index.query(
        vector=embeddings['embedding'],
        top_k=5,
        include_values=True
    )
    
    best_match = None
    highest_score = -1
    
    
    for result in query_results['matches']:
        product_id = result['id']
        row = df[df['id'] == product_id]
        score = result['score']
        if not row.empty and score > highest_score:
            best_match = row
            highest_score = score

    
    if best_match is not None and 'Answers' in best_match.columns and not best_match['Answers'].empty:
        return best_match['Answers'].values[0]
    else:
        return "Sorry, I couldn't find the answer to that. Please contact your local branch or call at 910-888-2341 for assistance."

def generate_response(user_query, end_conversation=False):
    global chat_history

    product_answer = query_pinecone(user_query)

    context_str = ""
    for entry in chat_history:
        context_str += f"Q: {entry['Question']}\nA: {entry['Answer']}\n"

    chat_history.append({"Question": user_query, "Answer": product_answer})

    
    inputs = {
        "context": context_str,
        "Question": user_query,
    }

    if end_conversation:
        chat_history = []  
        return "Anything else you want to know?"  

    
    return product_answer

template = """You are an intelligent and human-like assistant that answers customer queries related to TVS Credit services.
Your primary role is to help the customer by providing accurate, concise, and helpful answers to their questions.

### Database Information:
You have access to a database with common customer questions and their corresponding answers.
Whenever the customer asks a question, you should first check for the most similar question in the database and provide the answer accordingly.

If the customer asks something that is not explicitly covered in the database, use the context provided below to generate an appropriate response based on the available information.

### Context:
This is the conversation so far between you and the customer. Make sure to refer to this context when formulating your answer:
{context}

### Instructions:
1. **Use the context first**: Look at the previous conversation to ensure the response stays relevant to the ongoing discussion.
2. **Query Matching**: If the customer question matches any question in the database, provide the answer from the database’s `Answers` column.
3. **Handle Out-of-Scope Questions**: If you can't find a close match in the database, try your best to generate a response using the context and knowledge from prior conversations.
4. **Politeness and Clarity**: Always be polite, clear, and concise in your responses. If you're unsure about something, inform the customer politely.
5. **If the user asks you to summarize all the chat then you have to take important points from the chat_history using the question and answer and then make a summary of it and present it.**

### Answer the following:
Question: {Question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": RunnablePassthrough(), "Question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def detect_end_of_conversation(user_query):
    end_phrases = ["Thanks"]
    s = user_query.lower()
    s1 = s.split()
    for i in s1:
        if i in end_phrases:
            return True
    return False


def text_to_speech(text, language_code="en-US"):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    
    if language_code == "hi-IN":
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            name="hi-IN-Wavenet-A",  
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
    else: 
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",  
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    
    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_audio_file.name, "wb") as out:
        out.write(response.audio_content)

    return temp_audio_file.name


def process_transcription_step(transcription):
    if transcription is None or transcription.strip() == "":
        return "No transcription available."

    translated_text, original_language = translate_to_english(transcription)
    
    response = generate_response(translated_text)

    final_response = translate_to_original_language(response, original_language)

    return final_response



def process_transcription_and_generate_audio(transcription):
    text_response = process_transcription_step(transcription)

    translated_text, original_language = translate_to_english(transcription)

    language_code = "hi-IN" if original_language == "hi" else "en-US"

    audio_file_path = text_to_speech(text_response, language_code=language_code)

    return text_response, audio_file_path

def stop_audio():
    stop_playback()
    return "Playback stopped"



custom_css = """
<style>

    
    #submit_button {
        background-color: white; 
        border: none;
        color: white; 
        padding: 15px 15px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        width: auto;
        display: inline-block;
    }

    
    #process_button {
        background-color: white; 
        border: none;
        color: white; 
        padding: 15px 15px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        width: auto;
        display: inline-block;
    }

    
    #transcription_output, #response_output {
        font-size: 18px;
        color: #00ccff; 
        padding: 10px;
        border: 2px solid white; 
        border-radius: 5px;
        background-color: #067e3c; 
        width: 100%;
    }

    
    #mic_input {
        border: 2px solid white; 
        color: #ffffff;
        border-radius: 8px;
        width: 100%; 
        height: 200px; 
        background-color: #067e3c; 
        margin: 0 auto; 
        display: block;
        padding: 20px; 
    }

    
    #audio_output {
        border: 2px solid #ffcc00; 
        color: #ffffff;
        border-radius: 8px;
        width: 100%;
        height: 200px;
        background-color: #067e3c; 
    }

    
    #custom-container {
        background-color: #067e3c;
        border-radius: 10px;
        padding: 20px;
        color: #d6521c;
    }

    
    .container {
        max-width: 800px; 
        background-color: #0e3f70;
        border-radius: 10px;
        margin: 0 auto; 
        padding: 20px; 
    }
</style>
"""


with gr.Blocks() as app:
    
    gr.HTML(custom_css)
    with gr.Column(elem_classes="container"):
        gr.Markdown("# TVS Credit Assistant(T.A.R.U.N)")

        mic_input = gr.Audio(type="filepath", label="Record your query", elem_id="mic_input")

        with gr.Row():
            submit_button = gr.Button("Submit Audio", elem_id="submit_button")
            process_button = gr.Button("Process Transcription", elem_id="process_button")

        transcription_output = gr.Textbox(label="Transcription", elem_id="transcription_output")
        response_output = gr.Textbox(label="Response", elem_id="response_output")

        audio_output = gr.Audio(label="Response Audio", elem_id="audio_output")

        submit_button.click(transcribe_audio_step, inputs=[mic_input], outputs=transcription_output)

        process_button.click(process_transcription_and_generate_audio, inputs=[transcription_output], outputs=[response_output, audio_output])

if __name__ == "__main__":
    app.launch(share=True)