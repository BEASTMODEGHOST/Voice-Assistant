import gradio as gr
import os
import logging
from google.cloud import translate_v2 as translate
import tempfile
from groq import Groq
import speech_recognition as sr
from google.cloud import texttospeech
# Initialize Google Cloud Translate client


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"googlekey.json"  # for translation
translate_client = translate.Client()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"googlekey2.json"  # for text-to-speech
text_to_speech_client = texttospeech.TextToSpeechClient()




GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

translate_client = translate.Client()
text_to_speech_client = texttospeech.TextToSpeechClient()
# Initialize Groq API client (assuming it's already set up)

# Loan flow questions in both English and Hindi
loan_flow_questions_en = [
    "What is your age?",
    "Do you have a stable source of income?",
    "What is your monthly income?",
    "Do you want to proceed with the loan process?",
    "Please submit your ID proof, address proof, and other necessary documents.",
    "Would you like to know about loan repayment and amount options?",
    "Our interest rates range from 5% to 10%, based on your profile. Would you like more details?",
    "We'll verify your financial details and get back to you within 2-3 days.",
    "Loan approved! We’ll notify you via email or SMS. Thank you for using our service."
]

loan_flow_questions_hi = [
    "आपकी उम्र क्या है?",
    "क्या आपके पास स्थिर आय का स्रोत है?",
    "आपकी मासिक आय क्या है?",
    "क्या आप ऋण प्रक्रिया जारी रखना चाहते हैं?",
    "कृपया अपना पहचान प्रमाण, पता प्रमाण, और अन्य आवश्यक दस्तावेज़ जमा करें।",
    "क्या आप ऋण चुकौती और राशि विकल्पों के बारे में जानना चाहेंगे?",
    "हमारी ब्याज दरें आपकी प्रोफ़ाइल के आधार पर 5% से 10% तक होती हैं। क्या आप और विवरण चाहते हैं?",
    "हम आपके वित्तीय विवरणों की जांच करेंगे और 2-3 दिनों के भीतर आपको सूचित करेंगे।",
    "ऋण स्वीकृत! हम आपको ईमेल या एसएमएस के माध्यम से सूचित करेंगे। हमारी सेवा के लिए धन्यवाद।"
]

# Conversation flow state
conversation_flow = {"stage": 0, "language": "en"}

# Function to transcribe audio using Groq's API
def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file)
    return transcription.text

# Function to detect and translate Hindi to English if necessary
def translate_to_english(text):
    detection = translate_client.detect_language(text)
    detected_language = detection['language']

    if detected_language == "hi":  # If the text is in Hindi
        logging.info("Detected Hindi, translating to English...")
        translation = translate_client.translate(text, target_language="en")
        return translation['translatedText'], detected_language
    
    # If the text is already in English, return it as-is
    logging.info("Detected English, no translation needed.")
    return text, detected_language

# Function to translate English back to Hindi if required
def translate_to_original_language(text, target_language):
    translation = translate_client.translate(text, target_language=target_language)
    return translation['translatedText']
def google_text_to_speech(text, language_code="en-US"):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Choose voice parameters based on language
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Audio configuration for MP3 output
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Request to synthesize speech
    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )

    # Save the audio output to a temporary file
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_audio_file.name, "wb") as out:
        out.write(response.audio_content)

    return temp_audio_file.name
# Function to handle loan conversation based on the detected language
def loan_conversation_step(user_query):
    # Detect the language of the input query
    translated_query, detected_language = translate_to_english(user_query)
    
    if detected_language == "hi":  # If the detected language is Hindi
        loan_flow_questions = loan_flow_questions_hi
        conversation_flow["language"] = "hi"
    else:  # Default to English
        loan_flow_questions = loan_flow_questions_en
        conversation_flow["language"] = "en"
    
    # Process the conversation flow based on the stage and respond in the correct language
    if conversation_flow["stage"] == 0:
        conversation_flow["stage"] = 1
        return loan_flow_questions[0]
    
    elif conversation_flow["stage"] == 1:
        try:
            age = int(user_query)
            if age >= 21:
                conversation_flow["stage"] = 2
                return loan_flow_questions[1]
            else:
                conversation_flow["stage"] = 0
                return "I'm sorry, but you do not meet the minimum age requirement of 21 years for the loan." if conversation_flow["language"] == "en" else "मुझे खेद है, लेकिन आप ऋण के लिए न्यूनतम आयु आवश्यकता 21 वर्ष पूरी नहीं करते।"
        except ValueError:
            return "Please provide a valid age." if conversation_flow["language"] == "en" else "कृपया एक मान्य उम्र प्रदान करें।"
    
    elif conversation_flow["stage"] == 2:
        if "yes" in user_query.lower() or "हाँ" in user_query.lower():
            conversation_flow["stage"] = 3
            return loan_flow_questions[2]
        else:
            conversation_flow["stage"] = 0
            return "I'm sorry, but having a stable source of income is a mandatory requirement for the loan." if conversation_flow["language"] == "en" else "मुझे खेद है, लेकिन ऋण के लिए एक स्थिर आय स्रोत होना अनिवार्य आवश्यकता है।"
    
    elif conversation_flow["stage"] == 3:
        try:
            income = int(user_query.replace(",", ""))
            if income >= 15000:
                conversation_flow["stage"] = 4
                return loan_flow_questions[3]
            else:
                conversation_flow["stage"] = 0
                return "I'm sorry, but your income does not meet the minimum requirement for this loan." if conversation_flow["language"] == "en" else "मुझे खेद है, लेकिन आपकी आय इस ऋण के लिए न्यूनतम आवश्यकता को पूरा नहीं करती।"
        except ValueError:
            return "Please provide a valid income amount." if conversation_flow["language"] == "en" else "कृपया एक मान्य आय राशि प्रदान करें।"
    
    elif conversation_flow["stage"] == 4:
        if "yes" in user_query.lower() or "हाँ" in user_query.lower():
            conversation_flow["stage"] = 5
            return loan_flow_questions[4]
        else:
            conversation_flow["stage"] = 0
            return "Thank you for your time. Let us know if you need any assistance in the future." if conversation_flow["language"] == "en" else "आपके समय के लिए धन्यवाद। यदि आपको भविष्य में किसी सहायता की आवश्यकता हो तो हमें बताएं।"
    
    elif conversation_flow["stage"] == 5:
        conversation_flow["stage"] = 6
        return loan_flow_questions[5]
    
    elif conversation_flow["stage"] == 6:
        if "yes" in user_query.lower() or "हाँ" in user_query.lower():
            conversation_flow["stage"] = 7
            return loan_flow_questions[6]
        else:
            conversation_flow["stage"] = 0
            return "Thank you for your time. Let us know if you need any assistance in the future." if conversation_flow["language"] == "en" else "आपके समय के लिए धन्यवाद। यदि आपको भविष्य में किसी सहायता की आवश्यकता हो तो हमें बताएं।"
    
    elif conversation_flow["stage"] == 7:
        if "yes" in user_query.lower() or "हाँ" in user_query.lower():
            conversation_flow["stage"] = 8
            return loan_flow_questions[7]
        else:
            conversation_flow["stage"] = 0
            return "Thank you for your time. Let us know if you need any assistance in the future." if conversation_flow["language"] == "en" else "आपके समय के लिए धन्यवाद। यदि आपको भविष्य में किसी सहायता की आवश्यकता हो तो हमें बताएं।"
    
    elif conversation_flow["stage"] == 8:
        conversation_flow["stage"] = 9
        return loan_flow_questions[8]
    
    elif conversation_flow["stage"] == 9:
        conversation_flow["stage"] = 0
        return "Loan approved! We’ll notify you via email or SMS. Thank you for using our service." if conversation_flow["language"] == "en" else "ऋण स्वीकृत! हम आपको ईमेल या एसएमएस के माध्यम से सूचित करेंगे। हमारी सेवा के लिए धन्यवाद।"

# Gradio interface to handle speech and text conversation
def loan_bot(audio_input):
    # Recognize and transcribe user input using the Groq API
    user_transcription = transcribe_audio(audio_input)
    
    if not user_transcription:
        return "Sorry, I could not understand the audio.", None
    
    # Process the loan inquiry conversation flow
    bot_response = loan_conversation_step(user_transcription)
    # Determine the language for TTS (Text-to-Speech)
    if conversation_flow["language"] == "en":
        speech_output = google_text_to_speech(bot_response, language_code="en-US")
    else:
        speech_output = google_text_to_speech(bot_response, language_code="hi-IN")
    
    # Return the bot's response in both text and audio formats
    return bot_response, speech_output

# Create the Gradio interface
interface = gr.Interface(
    fn=loan_bot,
    inputs=gr.Audio(type="filepath", label="Record your query"),
    outputs=[gr.Textbox(label="Bot Response"), gr.Audio(label="Audio Response")],
    live=True,
    title="Loan Inquiry Bot",
    description="Speak your queries regarding loans, and the bot will respond with both text and voice in Hindi or English based on your query.",
)

interface.launch()