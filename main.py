from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import os
from gtts import gTTS
from datetime import datetime
import speech_recognition as sr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

PIXABAY_API_KEY = "50430149-79af3efd2415830b989abccc1"

STOP_WORDS = {"why", "are", "the", "is", "there", "what", "how", "do", "does", "a", "an", "for", "in", "on", "of"}

def extract_keywords(question):
    words = question.lower().split()
    keywords = [w for w in words if w not in STOP_WORDS]
    return " ".join(keywords) if keywords else question

def fetch_cartoon_image(question):
    keywords = extract_keywords(question)
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={keywords}&image_type=illustration&safesearch=true&per_page=3"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["hits"]:
            return data["hits"][0]["webformatURL"]
    return "https://cdn-icons-png.flaticon.com/512/616/616408.png"

def generate_tts(text):
    tts = gTTS(text=text, lang='en')
    filename = f"static/response_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    tts.save(filename)
    return filename

@app.post("/ask")
async def ask_question(audio_file: UploadFile = File(...)):
    # Save uploaded audio temporarily
    file_location = f"temp_{audio_file.filename}"
    with open(file_location, "wb") as f:
        f.write(await audio_file.read())

    # Speech recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_location) as source:
            audio_data = recognizer.record(source)
        question_text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except sr.RequestError:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail="Speech recognition service error")
    
    # Remove temp audio file
    os.remove(file_location)

    # Call Llama API with the recognized question text
    prompt = f"You are a fun and friendly teacher for 5-year-olds. Answer this question cheerfully in 2-3 simple sentences: {question_text}"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    result = response.json()
    explanation = result.get("response", "Hmm, I don't know the answer right now!")

    # Fetch image
    image_url = fetch_cartoon_image(question_text)

    # Generate TTS audio for explanation
    audio_path = generate_tts(explanation)
    audio_url = f"/static/{os.path.basename(audio_path)}"

    return {
        "question": question_text,
        "explanation": explanation.strip(),
        "image_url": image_url,
        "audio_url": audio_url,
    }
