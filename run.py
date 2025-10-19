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
import aiohttp

# .env 로드
load_dotenv()

# === 환경 변수 불러오기 ===
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
EL_KEY = os.getenv("EL_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
VOICE_ID = os.getenv("VOICE_ID", "MF3mGyEYCl7XYWbV9V6O") # 뒤에 굳이 한 번 더 쓰는 이유: .env 파일이 없거나, 해당 키가 빠졌거나, 서버 환경에서 환경변수가 설정되지 않은 경우의 "백업용 기본값"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PROMPT_GENERAL = os.getenv(
    "GEMINI_PROMPT_GENERAL",
    "Please answer in 1000 characters or less."
)
GEMINI_PROMPT_ANGEL = os.getenv(
    "GEMINI_PROMPT_ANGEL",
    "You are a kind, humble, and positive female streamer. Respond to the following message in a warm, encouraging, and respectful manner. Express empathy and gratitude, share uplifting experiences, and spread good vibes. If possible, go on a positive tangent that inspires or comforts your audience."
)
GEMINI_PROMPT_DEVIL = os.getenv(
    "GEMINI_PROMPT_DEVIL",
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

async def tts_play_async(message):
    if TTS_TYPE == "pyttsx3":
        await asyncio.to_thread(tts_play, message)  # 기존 동기 함수를 스레드로
    elif TTS_TYPE == "EL":
        await el_tts_async(message)

async def el_tts_async(message):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": EL_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": message[:2000],
        "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            print(f"🔎 ElevenLabs status: {resp.status}")
            content_type = resp.headers.get("Content-Type", "")
            print(f"🔎 Content-Type: {content_type}")

            content = await resp.read()

            # 🔍 실패 시 내용 출력
            if resp.status != 200:
                print("❌ ElevenLabs API Error:", content.decode(errors="ignore"))
                return  # 실패한 경우 pydub에 넘기지 않음

    def _play_audio():
        audio_content = AudioSegment.from_file(io.BytesIO(content), format="mp3")
        play(audio_content)

    await asyncio.to_thread(_play_audio)


# === 유틸: 디스코드 전송 길이 제한 처리 ===
DISCORD_LIMIT = 2000
SAFE_CHUNK = 1900  # 여유 버퍼

def chunk_message(text: str, limit: int = SAFE_CHUNK):
    chunks = []
    buf = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > limit:
            chunks.append(''.join(buf))
            buf = [line]
            size = len(line)
        else:
            buf.append(line)
            size += len(line)
    if buf:
        chunks.append(''.join(buf))
    # 혹시 한 줄이 엄청 길면(개행 없는 초장문) 스페이스 기준으로도 한 번 더 쪼갬
    fixed = []
    for c in chunks:
        if len(c) <= limit:
            fixed.append(c)
        else:
            words = c.split(' ')
            cur = []
            s = 0
            for w in words:
                if s + len(w) + 1 > limit:
                    fixed.append(' '.join(cur))
                    cur = [w]
                    s = len(w)
                else:
                    cur.append(w)
                    s += len(w) + 1
            if cur: fixed.append(' '.join(cur))
    return fixed


# === LLM (Gemini) ===
genai.configure(api_key=GEMINI_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)  # 전역 1회 초기화

def llm_sync(message: str, GEMINI_PROMPT: str) -> str:
    # 동기 SDK라면 executor에 넘길 함수
    response = MODEL.generate_content(f"{GEMINI_PROMPT + GEMINI_PROMPT_GENERAL}\n\n#########\n{message}\n#########\n")
    return getattr(response, "text", "") or ""


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
    if str(message.channel.id) != str(DISCORD_CHANNEL_ID):
        return

    print(f"\n[{message.author.name}]: {message.content}\n")
    async with message.channel.typing():
        loop = asyncio.get_running_loop()
        # LLM 호출을 스레드풀로 던짐(이벤트 루프 블로킹 방지)
        if message.content.startswith('/DEVIL'): response = await loop.run_in_executor(None, llm_sync, message.content[6:], GEMINI_PROMPT_DEVIL)
        else: response = await loop.run_in_executor(None, llm_sync, message.content, GEMINI_PROMPT_ANGEL)

    print(f"Response: {response}")

    # 2,000자 방지: 분할 전송
    parts = chunk_message(response)
    for part in parts:
        await message.channel.send(part)

    # TTS는 전송 이후 백그라운드로
    asyncio.create_task(tts_play_async(response))



if __name__ == "__main__":
    if TTS_TYPE == "pyttsx3":
        init_tts()
    print("\nRunning Discord Bot...\n")
    client.run(DISCORD_TOKEN)
