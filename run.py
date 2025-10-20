import os
import json
import asyncio
import tempfile
import uuid
import io
from collections import defaultdict
from dotenv import load_dotenv
import discord
from discord import FFmpegPCMAudio
import aiohttp
import websockets
import google.generativeai as genai

# === 환경 변수 로드 ===
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
EL_KEY = os.getenv("EL_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
VOICE_ID = os.getenv("VOICE_ID", "MF3mGyEYCl7XYWbV9V6O")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TTS_TYPE = os.getenv("TTS_TYPE", "EL")

# === 프롬프트 ===
GEMINI_PROMPT_ANGEL = os.getenv(
    "GEMINI_PROMPT_ANGEL",
    "You are a kind and warm female VTuber. Respond in a friendly and gentle way."
)
GEMINI_PROMPT_DEVIL = os.getenv(
    "GEMINI_PROMPT_DEVIL",
    "You are a mischievous, sarcastic VTuber. Respond playfully but with a sharp tone."
)
GEMINI_PROMPT_GENERAL = os.getenv(
    "GEMINI_PROMPT_GENERAL",
    "Keep responses under 1000 characters."
)

# === LLM 설정 ===
genai.configure(api_key=GEMINI_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)

# === 음성 잠금 ===
guild_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# === VTS 설정 ===
VTS_URL = "ws://localhost:8001"
VTS_PLUGIN_NAME = "Discord AI VTuber"
VTS_PLUGIN_AUTHOR = "Kibeom"


# ==============================================================
#                   VTS 인증 및 파라미터 제어
# ==============================================================

async def vts_connect_with_token():
    """vts_token.json의 토큰을 사용해 VTS 인증"""
    try:
        with open("vts_token.json", "r", encoding="utf-8") as f:
            token = json.load(f)["token"]
    except FileNotFoundError:
        print("vts_token.json 없음! 먼저 get_vts_token.py 실행해야 함.")
        return None

    ws = await websockets.connect(VTS_URL)
    auth_req = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth_token_login",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": VTS_PLUGIN_NAME,
            "pluginDeveloper": VTS_PLUGIN_AUTHOR,
            "authenticationToken": token
        }
    }

    await ws.send(json.dumps(auth_req))
    resp = json.loads(await ws.recv())

    if resp["data"].get("authenticated", False):
        print("VTS 인증 성공!")
        return ws
    else:
        print("VTS 인증 실패:", resp)
        return None


async def vts_set_parameter(ws, name, value):
    """VTS 파라미터 값 설정"""
    if ws is None:
        return
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"set_{name}",
        "messageType": "InjectParameterDataRequest",
        "data": {
            "faceFound": True,
            "mode": "set",
            "parameterValues": [
                {"id": name, "value": value, "weight": 1.0}
            ]
        }
    }
    await ws.send(json.dumps(msg))


async def set_expression(ws, mood):
    """표정 상태 전환"""
    expr = {
        "angel": {"MouthSmile": 1.0, "MouthOpen": 0.0},
        "devil": {"MouthSmile": 0.0, "MouthOpen": 0.2}
    }.get(mood, {})

    for k, v in expr.items():
        await vts_set_parameter(ws, k, v)


async def simulate_mouth(ws, duration=3.0):
    """입모양 움직임 시뮬레이션"""
    import random
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < duration:
        await vts_set_parameter(ws, "MouthOpen", random.uniform(0.1, 1.0))
        await asyncio.sleep(0.1)
    await vts_set_parameter(ws, "MouthOpen", 0.0)


# ==============================================================
#                  TTS (ElevenLabs)
# ==============================================================

async def el_tts_async(message):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": EL_KEY,
        "Content-Type": "application/json"
    }
    data = {"text": message[:2000], "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                print("ElevenLabs API Error:", await resp.text())
                return None
            return await resp.read()


async def play_mp3_bytes_in_discord(vc: discord.VoiceClient, mp3_bytes: bytes):
    if not mp3_bytes:
        return
    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    with open(tmp_path, "wb") as f:
        f.write(mp3_bytes)

    source = FFmpegPCMAudio(executable="ffmpeg", source=tmp_path, before_options="-nostdin", options="-vn")
    while vc.is_playing() or vc.is_paused():
        await asyncio.sleep(0.2)
    vc.play(source)
    while vc.is_playing():
        await asyncio.sleep(0.5)
    os.remove(tmp_path)


# ==============================================================
#                  Discord 봇 로직
# ==============================================================

def chunk_message(text: str, limit: int = 1900):
    chunks, buf, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > limit:
            chunks.append(''.join(buf))
            buf, size = [line], len(line)
        else:
            buf.append(line)
            size += len(line)
    if buf:
        chunks.append(''.join(buf))
    return chunks


genai.configure(api_key=GEMINI_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)

def llm_sync(message: str, prompt: str):
    response = MODEL.generate_content(f"{prompt + GEMINI_PROMPT_GENERAL}\n\n#########\n{message}\n#########\n")
    return getattr(response, "text", "") or ""


async def get_or_connect_voice_client(message):
    if not message.author.voice or not message.author.voice.channel:
        await message.channel.send("먼저 음성 채널에 들어와 줘!")
        return None
    voice_channel = message.author.voice.channel
    vc = discord.utils.get(client.voice_clients, guild=message.guild)
    if vc and vc.is_connected():
        if vc.channel != voice_channel:
            await vc.move_to(voice_channel)
        return vc
    return await voice_channel.connect()


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name='AI VTuber Online'))
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
        if message.content.startswith('/DEVIL'):
            response = await loop.run_in_executor(None, llm_sync, message.content[6:], GEMINI_PROMPT_DEVIL)
            mood = "devil"
        else:
            response = await loop.run_in_executor(None, llm_sync, message.content, GEMINI_PROMPT_ANGEL)
            mood = "angel"

    print(f"Response: {response}")

    # Discord 메시지 전송 (2000자 제한)
    for part in chunk_message(response):
        await message.channel.send(part)

    # === VTS 연동 + TTS 재생 ===
    ws = await vts_connect_with_token()
    if ws:
        await set_expression(ws, mood)

    lock = guild_locks[message.guild.id]
    async with lock:
        vc = await get_or_connect_voice_client(message)
        if vc:
            mp3_bytes = await el_tts_async(response)
            if mp3_bytes:
                if ws:
                    asyncio.create_task(simulate_mouth(ws, 4.0))
                await play_mp3_bytes_in_discord(vc, mp3_bytes)
                if vc.is_connected():
                    await vc.disconnect()

if __name__ == "__main__":
    print("\nRunning Discord AI VTuber...\n")
    client.run(DISCORD_TOKEN)
