import discord
import google.generativeai as genai
import json
import asyncio
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import pyttsx3

def initTTS():
    global engine

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        print(voice.id)
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)


def initVar():
    global EL_key
    global GEMINI_key
    global EL_voice
    global tts_type
    global GEMINI
    global EL
    global discord_token
    global discord_channel_id

    try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
    except:
        print("Unable to open JSON file.")
        exit()

    class GEMINI:
        key = data["keys"][0]["GEMINI_key"]
        model = data["GEMINI_data"][0]["model"]
        prompt = data["GEMINI_data"][0]["prompt"]

    class EL:
        key = data["keys"][0]["EL_key"]
        voice = data["EL_data"][0]["voice"]

    discord_token = data["keys"][0]["discord_token"]
    discord_channel_id = data["keys"][0]["discord_channel_id"]

    tts_type = "EL"

    if tts_type == "pyttsx3":
        initTTS()


def Controller_TTS(message):
    if tts_type == "EL":
        EL_TTS(message)
    elif tts_type == "pyttsx3":
        pyttsx3_TTS(message)


def pyttsx3_TTS(message):

    engine.say(message)
    engine.runAndWait()


def EL_TTS(message):

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{EL.voice}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': EL.key,
        'Content-Type': 'application/json'
    }
    data = {
        'text': message,
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    play(audio_content)


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if str(message.channel.id) == discord_channel_id:
        print(f"\n[{message.author.name}]- {message.content}\n")

        response = llm(message.content)
        print(response)
        await message.channel.send(response)
        Controller_TTS(response)

def llm(message):

    genai.configure(api_key=GEMINI.key)
    
    model = genai.GenerativeModel(GEMINI.model)

    response = model.generate_content(GEMINI.prompt + "\n\n#########\n" + message + "\n#########\n")

    return response.text


if __name__ == "__main__":
    initVar()
    print("\n\nRunning!\n\n")
    client.run(discord_token)
