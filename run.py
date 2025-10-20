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
import uuid, tempfile, os
from discord import FFmpegPCMAudio
from collections import defaultdict
guild_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


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
    engine.say(message)
    engine.runAndWait()

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
            content = await resp.read()
            if resp.status != 200:
                try:
                    print("ElevenLabs API Error:", content.decode(errors="ignore"))
                except:
                    pass
                return None
            return content

    #         print(f"ElevenLabs status: {resp.status}")
    #         content_type = resp.headers.get("Content-Type", "")
    #         print(f"Content-Type: {content_type}")

    #         content = await resp.read()

    #         # 실패 시 내용 출력
    #         if resp.status != 200:
    #             print("ElevenLabs API Error:", content.decode(errors="ignore"))
    #             return  # 실패한 경우 pydub에 넘기지 않음

    # def _play_audio():
    #     audio_content = AudioSegment.from_file(io.BytesIO(content), format="mp3")
    #     play(audio_content)

    # await asyncio.to_thread(_play_audio)

async def play_mp3_bytes_in_discord(vc: discord.VoiceClient, mp3_bytes: bytes):
    if not mp3_bytes:
        return

    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    try:
        with open(tmp_path, "wb") as f:
            f.write(mp3_bytes)

        source = FFmpegPCMAudio(
            executable="ffmpeg",   # PATH 잡혀 있으면 생략 가능
            source=tmp_path,
            before_options="-nostdin",
            options="-vn"
        )

        # 이미 재생 중이면 겹치지 않게 대기 (간단 큐 동작)
        while vc.is_playing() or vc.is_paused():
            await asyncio.sleep(0.2)

        vc.play(source)

        while vc.is_playing():
            await asyncio.sleep(0.5)

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

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

async def get_or_connect_voice_client(message) -> discord.VoiceClient | None:
    # 유저가 음성 채널에 없는 경우
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("먼저 음성 채널에 들어와 줘!")
        return None

    voice_channel = message.author.voice.channel
    # 이미 길드에 연결된 VC가 있으면 재사용
    vc = discord.utils.get(client.voice_clients, guild=message.guild)

    if vc and vc.is_connected():
        if vc.channel != voice_channel:
            await vc.move_to(voice_channel)  # 다른 채널이면 이동
        return vc

    # 없으면 새로 연결
    return await voice_channel.connect()


# === Discord 봇 ===
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name='VSCode로 개발 '))
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

    # TTS는 음성 채널로
    if TTS_TYPE == "EL":
        lock = guild_locks[message.guild.id]
        async with lock:
            vc = await get_or_connect_voice_client(message)
            if vc:
                mp3_bytes = await el_tts_async(response)
                if mp3_bytes:
                    await play_mp3_bytes_in_discord(vc, mp3_bytes)
                    # 상주시키고 싶지 않다면 재생 후 퇴장
                    if vc.is_connected():
                        await vc.disconnect()



if __name__ == "__main__":
    if TTS_TYPE == "pyttsx3":
        init_tts()
    print("\nRunning Discord Bot...\n")
    client.run(DISCORD_TOKEN)
