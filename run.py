import os
import sys
import re
import json
import asyncio
import tempfile
import uuid
from collections import defaultdict
from discord import FFmpegOpusAudio

from dotenv import load_dotenv
import discord
import aiohttp
import websockets
import google.generativeai as genai
import time

# =========================
# 경로/환경
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DISCORD_TOKEN       = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID  = os.getenv("DISCORD_CHANNEL_ID")  # 텍스트 채널 ID
EL_KEY              = os.getenv("EL_KEY")
GEMINI_KEY          = os.getenv("GEMINI_KEY")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VOICE_ID_FALLBACK   = os.getenv("VOICE_ID", "MF3mGyEYCl7XYWbV9V6O")
FFMPEG_PATH         = os.getenv("FFMPEG_PATH", "ffmpeg")

# VTS
VTS_URL           = os.getenv("VTS_URL", "ws://localhost:8001")
VTS_ENABLED       = os.getenv("VTS_ENABLED", "true").lower() == "true"
VTS_PLUGIN_NAME   = "Discord AI VTuber"
VTS_PLUGIN_AUTHOR = "Kibeom"
VTS_TOKEN_PATH    = os.path.join(BASE_DIR, "vts_token.json")


# Persona 파일 경로
REI_JSON_PATH = os.path.join(BASE_DIR, "persona", "configs", "rei.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.getenv("MEMORY_DIR", BASE_DIR)

# =========================
# persona 모듈 임포트 (패키지/폴더 모두 대응)
# =========================
try:
    from persona.persona_manager import PersonaManager
    from persona.memory_store import MemoryStore
except ModuleNotFoundError:
    sys.path.append(os.path.join(BASE_DIR, "persona"))
    from persona_manager import PersonaManager  # type: ignore
    from memory_store import MemoryStore       # type: ignore

# =========================
# LLM 설정 + JSON 파서 (reply만 사용)
# =========================
genai.configure(api_key=GEMINI_KEY)
MODEL = genai.GenerativeModel(GEMINI_MODEL)

def _extract_first_json(text: str):
    """
    모델 출력에서 첫 번째 JSON 객체만 안전하게 추출해 dict로 반환.
    - ```json ... ``` 코드펜스도 처리
    - 중괄호 균형을 스택으로 검사(문자열/이스케이프 고려)
    실패 시 None.
    """
    if not text:
        return None

    cand = text

    # 코드펜스 처리: ```json ... ``` or ``` ... ```
    if "```" in cand:
        i = cand.find("```")
        j = cand.find("```", i + 3)
        if j != -1:
            cand = cand[i + 3 : j]
    # 펜스 안의 선행 'json' 토큰 제거
    cand = cand.lstrip()
    if cand.lower().startswith("json"):
        cand = cand[4:].lstrip()

    # 스택으로 첫 번째 JSON 객체 찾기
    start = cand.find("{")
    while start != -1:
        stack = 0
        in_str = False
        escape = False
        for idx in range(start, len(cand)):
            ch = cand[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":  # 문자열 내 이스케이프
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    stack += 1
                elif ch == "}":
                    stack -= 1
                    if stack == 0:
                        segment = cand[start : idx + 1]
                        try:
                            return json.loads(segment)
                        except Exception:
                            # 이 블록이 JSON이 아니면 다음 '{'부터 다시 시도
                            break
        # 다음 후보 '{'로 이동
        start = cand.find("{", start + 1)
    return None

def llm_json_sync(prompt: str):
    guard = (
        'Output strictly as JSON with keys "reply","mood_delta","affinity_delta". '
        'No markdown, no prose. Example: {"reply":"...","mood_delta":0,"affinity_delta":0}'
    )
    resp = MODEL.generate_content(prompt + "\n\n" + guard)
    raw = getattr(resp, "text", "") or ""

    data = _extract_first_json(raw)
    if isinstance(data, dict):
        data.setdefault("reply", "")
        data.setdefault("mood_delta", 0)
        data.setdefault("affinity_delta", 0)
        if isinstance(data["reply"], (dict, list)):
            data["reply"] = json.dumps(data["reply"], ensure_ascii=False)
        return data

    # 파싱 실패: 원문을 잘라 reply로만 사용
    return {"reply": raw[:500], "mood_delta": 0, "affinity_delta": 0}

# =========================
# Persona 초기화
# =========================
memory = MemoryStore(base_dir=MEMORY_DIR, stm_maxlen=10)
persona_manager = PersonaManager(REI_JSON_PATH, memory, base_dir=BASE_DIR)

# =========================
# VTS 연결/제어
# =========================
_last_vts_log = 0.0

async def vts_connect_with_token(retries: int = 1, timeout_sec: float = 2.0):
    """
    VTS가 꺼져 있거나 포트가 다르면 None 반환하고 그냥 진행.
    연결 실패는 on_message를 중단시키지 않음.
    """
    global _last_vts_log
    if not VTS_ENABLED:
        return None

    try:
        with open(VTS_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = json.load(f)["token"]
    except FileNotFoundError:
        now = time.time()
        if now - _last_vts_log > 10:
            print("vts_token.json 없음. VTS 연동을 건너뜁니다.")
            _last_vts_log = now
        return None

    for attempt in range(1, retries + 1):
        try:
            ws = await asyncio.wait_for(websockets.connect(VTS_URL), timeout=timeout_sec)
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
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_sec))
            if resp.get("data", {}).get("authenticated"):
                print("VTS 인증 성공")
                return ws
            else:
                now = time.time()
                if now - _last_vts_log > 10:
                    print(f"VTS 인증 실패: {resp}")
                    _last_vts_log = now
                try:
                    await ws.close()
                except:
                    pass
                return None
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            if attempt >= retries:
                now = time.time()
                if now - _last_vts_log > 10:
                    print(f"VTS 연결 실패(무시하고 진행): {e}")
                    _last_vts_log = now
            await asyncio.sleep(0.2)
        except Exception as e:
            now = time.time()
            if now - _last_vts_log > 10:
                print(f"VTS 예외(무시하고 진행): {e}")
                _last_vts_log = now
            return None
    return None


async def vts_set_parameter(ws, name, value):
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
            "parameterValues": [{"id": name, "value": float(value), "weight": 1.0}]
        }
    }
    await ws.send(json.dumps(msg))

async def vts_apply_params(ws, params: dict):
    if not ws or not params:
        return
    for k, v in params.items():
        await vts_set_parameter(ws, k, v)

async def simulate_mouth(ws, duration=3.5):
    """간단한 입모양 동기화 (고정 길이 시뮬)"""
    import random, time
    if ws is None:
        await asyncio.sleep(duration)
        return
    start = time.monotonic()
    while time.monotonic() - start < duration:
        await vts_set_parameter(ws, "MouthOpen", random.uniform(0.1, 1.0))
        await asyncio.sleep(0.1)
    await vts_set_parameter(ws, "MouthOpen", 0.0)

# =========================
# ElevenLabs TTS (reply만 재생)
# =========================
async def ensure_bot_voice_flags(vc: discord.VoiceClient):
    try:
        me = vc.guild.me  # discord.Member
        vs = me.voice
        if vs and (vs.self_deaf or vs.self_mute):
            # 권한 없으면 무시됨
            await me.edit(deafen=False, mute=False)
    except Exception as e:
        print(f"봇 음소거/데프 해제 실패(무시 가능): {e}")

async def el_tts_async(message, voice_id: str):
    vid = voice_id if voice_id and "<" not in voice_id and ">" not in voice_id else VOICE_ID_FALLBACK
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {"accept": "audio/mpeg", "xi-api-key": EL_KEY, "Content-Type": "application/json"}
    data = {"text": str(message)[:2000], "voice_settings": {"stability": 0.75, "similarity_boost": 0.75}}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"ElevenLabs API Error [{resp.status}]: {body}")
                return None
            mp3 = await resp.read()
            if not mp3:
                print("TTS 결과가 비어 있음")
            return mp3

async def play_mp3_bytes_in_discord(vc: discord.VoiceClient, mp3_bytes: bytes, volume: float = 1.0):
    if not mp3_bytes:
        return

    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    with open(tmp_path, "wb") as f:
        f.write(mp3_bytes)

    try:
        await ensure_bot_voice_flags(vc)

        # 이전 재생 종료 대기
        while vc.is_playing() or vc.is_paused():
            await asyncio.sleep(0.1)

        # ffmpeg로 mp3 => opus(48kHz, stereo). -filter:a 로 볼륨 가산 가능
        # volume 인자는 0.0~2.0 정도 권장
        vol = max(0.0, float(volume))
        opus_src = FFmpegOpusAudio(
            source=tmp_path,
            executable=FFMPEG_PATH,
            bitrate=128,  # kbps
            before_options="-nostdin",
            options=f'-vn -ac 2 -ar 48000 -filter:a "volume={vol}"'
        )

        print("음성 재생 시작 (Opus)")
        vc.play(opus_src)

        while vc.is_playing():
            await asyncio.sleep(0.2)

        print("음성 재생 종료")
    except FileNotFoundError:
        print("fmpeg 를 찾지 못했습니다. .env의 FFMPEG_PATH를 절대경로로 지정하거나 ffmpeg를 설치하세요.")
    except Exception as e:
        print(f"음성 재생 오류: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# =========================
# Discord 봇
# =========================
guild_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

def chunk_message(text: str, limit: int = 1900):
    chunks, buf, size = [], [], 0
    for line in str(text).splitlines(keepends=True):
        if size + len(line) > limit:
            chunks.append(''.join(buf)); buf, size = [line], len(line)
        else:
            buf.append(line); size += len(line)
    if buf: chunks.append(''.join(buf))
    return chunks

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
    await client.change_presence(activity=discord.Game(name='AI VTuber (Persona Mode)'))
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
        def _call_llm(prompt):  # PersonaManager가 기대하는 콜백
            return llm_json_sync(prompt)
        guild_id = message.guild.id if message.guild else None
        result = await loop.run_in_executor(
            None, persona_manager.step, str(message.author.id), guild_id, message.content, _call_llm
        )

    # === 오직 reply만 사용 ===
    response  = str(result.get("reply", ""))
    mood      = result.get("mood", "neutral")
    voice_id  = result.get("voice_id")
    vts_params= result.get("vts_params", {})
    print(f"[Persona] mood={mood}, voice={voice_id} | reply={response[:80]}...")

    # 텍스트 채팅: reply만 전송 (2000자 제한 대응)
    for part in chunk_message(response):
        await message.channel.send(part)

    # VTS 표정 적용 + 음성으로 reply 말하기
    ws = None
    try:
        ws = await vts_connect_with_token()
        if ws:
            await vts_apply_params(ws, vts_params)
    except Exception as e:
        # 어떤 이유든 VTS 실패는 무시하고 계속 진행
        print(f"VTS 사용 안 함(오류 무시): {e}")
        ws = None


    lock = guild_locks[message.guild.id]
    async with lock:
        vc = await get_or_connect_voice_client(message)
        if vc:
            mp3_bytes = await el_tts_async(response, voice_id)
            if mp3_bytes:
                if ws:
                    # 길이 추정이 어려우니 고정 4초 정도 입모양 시뮬
                    asyncio.create_task(simulate_mouth(ws, 4.0))
                await play_mp3_bytes_in_discord(vc, mp3_bytes, volume=1.4)
                # if vc.is_connected():
                #     await vc.disconnect()

if __name__ == "__main__":
    print("\nRunning Discord AI VTuber (Persona Mode) ...\n")
    client.run(DISCORD_TOKEN)
