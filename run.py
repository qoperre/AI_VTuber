import os
from dotenv import load_dotenv
import discord
import google.generativeai as genai
import asyncio
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import pyttsx3

# .env 로드
load_dotenv()

# === 환경 변수 불러오기 ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
EL_KEY = os.getenv("EL_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
VOICE_ID = os.getenv("VOICE_ID", "MF3mGyEYCl7XYWbV9V6O")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_PROMPT = os.getenv(
    "GEMINI_PROMPT",
    "You are a toxic, entitled, evil female streamer. Respond to the following message in a toxic and rude manner."
)
TTS_TYPE = os.getenv("TTS_TYPE", "EL")

# === TTS 초기화 ===
def init_tts():
    global engine
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.setProperty("volume", 1.0)

def tts_play(message):
    if TTS_TYPE == "pyttsx3":
        engine.say(message)
        engine.runAndWait()
    elif TTS_TYPE == "EL":
        el_tts(message)

def el_tts(message):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": EL_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": message,
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
    }
    response = requests.post(url, headers=headers, json=data)
    audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    play(audio_content)

# === LLM (Gemini) ===
def llm(message):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(f"{GEMINI_PROMPT}\n\n#########\n{message}\n#########\n")
    return response.text

# === Discord 봇 ===
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if str(message.channel.id) == str(DISCORD_CHANNEL_ID):
        print(f"\n[{message.author.name}]: {message.content}\n")
        response = llm(message.content)
        print(f"Response: {response}")
        await message.channel.send(response)
        tts_play(response)

if __name__ == "__main__":
    if TTS_TYPE == "pyttsx3":
        init_tts()
    print("\nRunning Discord Bot...\n")
    client.run(DISCORD_TOKEN)
