import os
import sys
import re
import json
import asyncio
import tempfile
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
from io import BytesIO
from discord import FFmpegPCMAudio
import discord.abc

from dotenv import load_dotenv
import discord
import websockets
import google.generativeai as genai
import time
import wave
import numpy as np
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore
try:
    from melo.api import TTS as MeloTTS  # type: ignore
except Exception:
    MeloTTS = None  # type: ignore
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    import discord.sinks as discord_sinks  # type: ignore
except Exception:
    discord_sinks = None  # type: ignore

# =========================
# 경로/환경
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DISCORD_TOKEN       = os.getenv("DISCORD_TOKEN")
DISCORD_CHANNEL_ID  = os.getenv("DISCORD_CHANNEL_ID")  # 텍스트 채널 ID
GEMINI_KEY          = os.getenv("GEMINI_KEY")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FFMPEG_PATH         = os.getenv("FFMPEG_PATH", "ffmpeg")

VOICE_LISTEN_ENABLED      = os.getenv("VOICE_LISTEN_ENABLED", "true").lower() == "true"
VOICE_LISTEN_WINDOW_SEC   = float(os.getenv("VOICE_LISTEN_WINDOW_SEC", "4.0"))
VOICE_TRANSCRIPT_MIN_CHARS= int(os.getenv("VOICE_TRANSCRIPT_MIN_CHARS", "4"))
VOICE_TRANSCRIPT_MAX_CHARS= int(os.getenv("VOICE_TRANSCRIPT_MAX_CHARS", "200"))
VOICE_PLAYBACK_VOLUME     = float(os.getenv("VOICE_PLAYBACK_VOLUME", "1.4"))

# MeloTTS (Japanese TTS) + Whisper STT defaults
TTS_PROVIDER          = os.getenv("TTS_PROVIDER", "MELO").upper()  # MELO or GEMINI
MELO_TTS_LANGUAGE     = os.getenv("MELO_TTS_LANGUAGE", "JP")
MELO_TTS_SPEAKER      = os.getenv("MELO_TTS_SPEAKER", "ja-JP-NanamiNeural")
MELO_TTS_DEVICE       = os.getenv("MELO_TTS_DEVICE", "auto")
MELO_TTS_AUDIO_FORMAT = os.getenv("MELO_TTS_AUDIO_FORMAT", "audio/wav")

STT_PROVIDER          = os.getenv("STT_PROVIDER", "WHISPER").upper()
WHISPER_MODEL         = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE        = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_LANGUAGE      = os.getenv("WHISPER_LANGUAGE", "ko")
WHISPER_COMPUTE_TYPE  = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

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

_whisper_model: Optional[Any] = None
_whisper_model_lock = asyncio.Lock()
_melo_tts_model: Optional[Any] = None
_melo_tts_lock = asyncio.Lock()

VOICE_RECEIVE_SUPPORTED = bool(discord_sinks) and hasattr(discord.VoiceClient, "start_recording")
_voice_receive_warned = False
voice_listener_tasks: dict[int, asyncio.Task] = {}
voice_listener_channels: dict[int, discord.TextChannel] = {}

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
# Google Cloud / Gemini TTS
# =========================
async def ensure_bot_voice_flags(vc: discord.VoiceClient):
    try:
        me = vc.guild.me  # discord.Member
        vs = me.voice
        if vs and (vs.self_deaf or vs.self_mute):
            await me.edit(deafen=False, mute=False)
    except Exception as e:
        print(f"Voice state adjustment failed (ignored): {e}")


def _normalize_voice_config(voice_cfg: Dict[str, Any]) -> Dict[str, Any]:
    speaker = str(
        voice_cfg.get("voice_name")
        or voice_cfg.get("voice_id")
        or MELO_TTS_SPEAKER
        or ""
    ).strip()
    if not speaker:
        speaker = MELO_TTS_SPEAKER

    language_code = voice_cfg.get("language_code") or MELO_TTS_LANGUAGE or "JP"
    language_code = str(language_code).strip()

    speaking_rate = float(voice_cfg.get("speaking_rate", 1.0))
    pitch = float(voice_cfg.get("pitch", 0.0))

    return {
        "speaker": speaker,
        "language_code": language_code,
        "speaking_rate": speaking_rate,
        "pitch": pitch,
        "raw": voice_cfg,
    }


def _resolve_speaker_id(model: Any, speaker: Any) -> Any:
    if isinstance(speaker, (int, float)):
        return int(speaker)
    if isinstance(speaker, str):
        s = speaker.strip()
        if s.isdigit():
            return int(s)
        slug = re.sub(r"[^a-z0-9]", "", s.lower())
        candidate = None
        spk_map = None
        hps = getattr(model, "hps", None)
        data = getattr(hps, "data", None)
        if isinstance(data, dict):
            spk_map = data.get("spk2id")
        elif hasattr(data, "spk2id"):
            spk_map = getattr(data, "spk2id")
        if isinstance(spk_map, dict):
            for key, val in spk_map.items():
                key_slug = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if not key_slug:
                    continue
                if slug == key_slug or slug in key_slug or key_slug in slug:
                    candidate = val
                    break
            if candidate is not None:
                if isinstance(candidate, (int, float)):
                    return int(candidate)
                if isinstance(candidate, str) and candidate.isdigit():
                    return int(candidate)
        return 0
    return 0


def _resolve_device(preferred: str) -> str:
    pref = (preferred or "").lower()
    if pref in ("", "auto"):
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return preferred


async def _ensure_melo_model(language_code: str):
    global _melo_tts_model
    if _melo_tts_model is not None:
        return _melo_tts_model
    async with _melo_tts_lock:
        if _melo_tts_model is None:
            def _load():
                if MeloTTS is None:
                    raise RuntimeError("MeloTTS library not installed. Run `pip install melotts`.")
                device = _resolve_device(MELO_TTS_DEVICE)
                lang = (language_code or MELO_TTS_LANGUAGE or "JP").upper()
                return MeloTTS(language=lang, device=device)
            loop = asyncio.get_running_loop()
            _melo_tts_model = await loop.run_in_executor(None, _load)
    return _melo_tts_model


async def melo_tts_async(text: str, voice_cfg: Dict[str, Any]) -> Optional[Tuple[bytes, str]]:
    info = _normalize_voice_config(voice_cfg)
    try:
        model = await _ensure_melo_model(info["language_code"])
    except Exception as e:
        print(f"MeloTTS init failed: {e}")
        return None

    speaker_raw = info.get("speaker") or MELO_TTS_SPEAKER
    speaker_id = _resolve_speaker_id(model, speaker_raw)
    text_input = str(text)[:5000]
    language_hint = info["language_code"] or MELO_TTS_LANGUAGE or "JP"
    default_sr = getattr(model, "sample_rate", 44100)

    def _generate() -> Optional[bytes]:
        if MeloTTS is None:
            raise RuntimeError("MeloTTS library not available")

        sr = default_sr
        audio_arr: Optional[Any] = None

        if hasattr(model, "tts"):
            try:
                result = model.tts(text_input, speaker_id=speaker_id, language=language_hint)  # type: ignore[attr-defined]
            except TypeError:
                try:
                    result = model.tts(text_input, speaker_id=speaker_id)  # type: ignore[attr-defined]
                except TypeError:
                    result = model.tts(text_input)  # type: ignore[attr-defined]

            if isinstance(result, (bytes, bytearray)):
                return bytes(result)
            if isinstance(result, str):
                candidate_path = result.strip()
                if candidate_path and os.path.exists(candidate_path):
                    with open(candidate_path, "rb") as f:
                        return f.read()
                else:
                    tmp_path = os.path.join(tempfile.gettempdir(), f"melotts_{uuid.uuid4().hex}.wav")
                    try:
                        model.tts_to_file(text_input, speaker_id=speaker_id, language=language_hint, audio_path=tmp_path)  # type: ignore[attr-defined]
                        if os.path.exists(tmp_path):
                            with open(tmp_path, "rb") as f:
                                return f.read()
                    except Exception:
                        pass
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

            if isinstance(result, dict):
                if "audio" in result:
                    audio_arr = result.get("audio")
                if "sample_rate" in result:
                    try:
                        sr = int(result.get("sample_rate", sr))
                    except Exception:
                        pass
            elif isinstance(result, tuple) and len(result) >= 2:
                first, second = result[0], result[1]
                if isinstance(first, (int, float)):
                    sr = int(first)
                    audio_arr = second
                else:
                    audio_arr = first
                    if isinstance(second, (int, float)):
                        sr = int(second)
            elif result is not None:
                audio_arr = result

            if audio_arr is not None:
                if torch is not None and isinstance(audio_arr, torch.Tensor):
                    audio_arr = audio_arr.detach().cpu().numpy()
                elif isinstance(audio_arr, (list, tuple)):
                    audio_arr = np.asarray(audio_arr)
                elif not isinstance(audio_arr, np.ndarray):
                    try:
                        audio_arr = np.asarray(audio_arr)
                    except Exception:
                        audio_arr = None

            if isinstance(audio_arr, np.ndarray):
                audio_arr = audio_arr.astype(np.float32)
                if audio_arr.ndim == 1:
                    audio_arr = audio_arr[:, None]
                if audio_arr.size > 0 and audio_arr.dtype.kind != 'U':
                    audio_arr = np.clip(audio_arr, -1.0, 1.0)
                    pcm16 = (audio_arr * 32767.0).astype(np.int16)
                    buf = BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(pcm16.shape[1])
                        wf.setsampwidth(2)
                        wf.setframerate(int(sr or default_sr))
                        wf.writeframes(pcm16.tobytes())
                    return buf.getvalue()

        tmp_path = os.path.join(tempfile.gettempdir(), f"melotts_{uuid.uuid4().hex}.wav")
        try:
            success = False
            func = getattr(model, "tts_to_file", None)
            if callable(func):
                attempts = (
                    {"speaker_id": speaker_id, "language": language_hint, "audio_path": tmp_path},
                    {"speaker_id": speaker_id, "audio_path": tmp_path},
                    {"speaker": speaker_id, "audio_path": tmp_path},
                    {"speaker_id": speaker_id, "language": language_hint, "output_path": tmp_path},
                    {"speaker_id": speaker_id, "output_path": tmp_path},
                    {"speaker_id": speaker_id, "language": language_hint, "output_file": tmp_path},
                    {"speaker_id": speaker_id, "output_file": tmp_path},
                )
                for kwargs in attempts:
                    try:
                        func(text_input, **{k: v for k, v in kwargs.items() if v is not None})
                        success = True
                        break
                    except TypeError:
                        continue
                    except Exception as e:
                        print(f"MeloTTS tts_to_file attempt failed: {e}")
            if not success and hasattr(model, "save_wav"):
                func = getattr(model, "save_wav")
                attempts = (
                    {"path": tmp_path, "speaker_id": speaker_id, "language": language_hint},
                    {"path": tmp_path, "speaker_id": speaker_id},
                    {"path": tmp_path, "speaker": speaker_id},
                )
                for kwargs in attempts:
                    try:
                        func(text_input, **{k: v for k, v in kwargs.items() if v is not None})
                        success = True
                        break
                    except TypeError:
                        continue
                    except Exception as e:
                        print(f"MeloTTS save_wav attempt failed: {e}")
            if not success:
                return None
            if not os.path.exists(tmp_path):
                return None
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return None

    loop = asyncio.get_running_loop()
    try:
        audio_bytes = await loop.run_in_executor(None, _generate)
    except Exception as e:
        print(f"MeloTTS synthesis failed: {e}")
        audio_bytes = None

    if not audio_bytes:
        print("MeloTTS returned empty audio.")
        return None
    return audio_bytes, (MELO_TTS_AUDIO_FORMAT or "audio/wav")
async def synthesize_speech_async(message: str, voice_cfg: Optional[Dict[str, Any]] = None) -> Optional[Tuple[bytes, str]]:
    text = (message or "").strip()
    if not text:
        return None
    voice_cfg = voice_cfg or {}
    if TTS_PROVIDER == "MELO":
        audio = await melo_tts_async(text, voice_cfg)
        if audio:
            return audio
    elif TTS_PROVIDER == "GEMINI":
        print("Gemini audio provider not implemented; falling back to MeloTTS.")
        audio = await melo_tts_async(text, voice_cfg)
        if audio:
            return audio
    else:
        audio = await melo_tts_async(text, voice_cfg)
        if audio:
            return audio
    return None


async def _ensure_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_model_lock:
        if _whisper_model is None:
            def _load():
                if whisper is None:
                    raise RuntimeError("openai-whisper is not installed. Run `pip install openai-whisper`.")
                device = _resolve_device(WHISPER_DEVICE)
                try:
                    return whisper.load_model(WHISPER_MODEL, device=device)
                except TypeError:
                    model = whisper.load_model(WHISPER_MODEL)
                    if device != "cpu" and hasattr(model, "to"):
                        model.to(device)
                    return model
            loop = asyncio.get_running_loop()
            _whisper_model = await loop.run_in_executor(None, _load)
    return _whisper_model


def _whisper_language(code: Optional[str]) -> str:
    lang = (code or WHISPER_LANGUAGE or "ko").strip()
    if "-" in lang:
        lang = lang.split("-")[0]
    return lang.lower()


def _whisper_use_fp16(device: str) -> bool:
    dev = device.lower()
    if dev in ("cuda", "gpu"):
        return torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
    return False


async def transcribe_audio_async(audio_bytes: bytes, *, sample_rate: int = 48000, language_code: Optional[str] = None, audio_channel_count: int = 2) -> str:
    if not audio_bytes:
        return ""

    if STT_PROVIDER == "WHISPER":
        try:
            model = await _ensure_whisper_model()
        except Exception as e:
            print(f"Whisper model init failed: {e}")
        else:
            tmp_path = os.path.join(tempfile.gettempdir(), f"whisper_{uuid.uuid4().hex}.wav")
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)

            language = _whisper_language(language_code)
            device = _resolve_device(WHISPER_DEVICE)
            fp16 = _whisper_use_fp16(device)

            def _transcribe(path: str) -> str:
                result = model.transcribe(path, language=language, task="transcribe", fp16=fp16)
                if not result:
                    return ""
                return str(result.get("text", "")).strip()

            loop = asyncio.get_running_loop()
            try:
                text = await loop.run_in_executor(None, lambda: _transcribe(tmp_path))
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if text:
                return text.strip()

    return ""


async def process_voice_sink(sink: Any, vc: discord.VoiceClient, text_channel: discord.TextChannel):
    print(f"[DEBUG] Sink audio_data keys: {list(getattr(sink, 'audio_data', {}).keys())}")
    
    guild = vc.guild
    if not guild:
        return

    for user_id, audio in getattr(sink, "audio_data", {}).items():
        if not audio or user_id == getattr(client.user, "id", None):
            continue
        member = guild.get_member(user_id)
        if member is None:
            continue

        file_obj = getattr(audio, "file", None)
        if file_obj is None:
            continue
        try:
            file_obj.seek(0)
        except Exception:
            pass

        if hasattr(file_obj, "read"):
            audio_bytes = file_obj.read()
            try:
                file_obj.seek(0)
            except Exception:
                pass
        elif hasattr(file_obj, "getvalue"):
            audio_bytes = file_obj.getvalue()
        else:
            continue

        sample_rate = getattr(audio, "sample_rate", 48000) or 48000
        channels = getattr(audio, "channels", getattr(audio, "channel_count", 2)) or 2

        transcript = await transcribe_audio_async(
            audio_bytes,
            sample_rate=int(sample_rate),
            audio_channel_count=int(channels),
            language_code=WHISPER_LANGUAGE,
        )

        if transcript:
            print(f"[STT DEBUG] Whisper recognized: \"{transcript}\"")
        else:
            print("[STT DEBUG] Whisper returned no text")

        if not transcript:
            continue
        transcript = transcript.strip()
        if len(transcript) < VOICE_TRANSCRIPT_MIN_CHARS:
            continue
        if len(transcript) > VOICE_TRANSCRIPT_MAX_CHARS:
            transcript = transcript[:VOICE_TRANSCRIPT_MAX_CHARS].rstrip() + "..."

        await handle_user_input(member, text_channel, transcript, source="voice", announce=True)


async def voice_listener_loop(guild_id: int, vc: discord.VoiceClient):
    global voice_listener_tasks
    try:
        while VOICE_LISTEN_ENABLED and VOICE_RECEIVE_SUPPORTED and vc.is_connected():
            text_channel = voice_listener_channels.get(guild_id)
            if text_channel is None:
                await asyncio.sleep(1.0)
                continue

            if not discord_sinks:
                break

            sink = discord_sinks.WaveSink()  # type: ignore[attr-defined]
            finished = asyncio.Event()

            async def _once_done(_sink, *_) -> None:
                if not finished.is_set():
                    finished.set()

            try:
                vc.start_recording(sink, _once_done)
            except Exception as e:
                print(f"[VoiceLoop WARN] start_recording failed: {e}")
                await asyncio.sleep(1.0)
                continue

            try:
                await asyncio.sleep(max(1.0, VOICE_LISTEN_WINDOW_SEC))
            except asyncio.CancelledError:
                vc.stop_recording()
                await finished.wait()
                if hasattr(sink, "cleanup"):
                    try:
                        sink.cleanup()
                    except Exception:
                        pass
                raise

            vc.stop_recording()
            await finished.wait()

            try:
                await process_voice_sink(sink, vc, text_channel)
            except Exception as e:
                print(f"[VoiceLoop WARN] Voice processing error: {e}")
            finally:
                if hasattr(sink, "cleanup"):
                    try:
                        sink.cleanup()
                    except Exception as e:
                        print(f"[VoiceLoop WARN] Sink cleanup skipped: {e}")

    except Exception as loop_error:
        print(f"[VoiceLoop ERROR] Loop crashed: {loop_error} (will auto-restart)")
        await asyncio.sleep(1.0)
        asyncio.create_task(voice_listener_loop(guild_id, vc))  # 🔁 자동 재시작
    finally:
        voice_listener_tasks.pop(guild_id, None)
        voice_listener_channels.pop(guild_id, None)



async def ensure_voice_listener(vc: discord.VoiceClient, text_channel: discord.TextChannel):
    global _voice_receive_warned
    if not VOICE_LISTEN_ENABLED:
        return
    if not VOICE_RECEIVE_SUPPORTED:
        if not _voice_receive_warned:
            print("Voice receive not supported (discord.sinks missing or outdated discord.py). Voice chat recognition disabled.")
            _voice_receive_warned = True
        return

    guild = vc.guild
    if not guild:
        return
    guild_id = guild.id
    voice_listener_channels[guild_id] = text_channel

    task = voice_listener_tasks.get(guild_id)
    if task and not task.done():
        return

    voice_listener_tasks[guild_id] = asyncio.create_task(voice_listener_loop(guild_id, vc))


async def handle_user_input(user: discord.abc.User, channel: discord.abc.Messageable, content: str, *, source: str = "text", announce: bool = False):
    content = (content or "").strip()
    if not content:
        return

    display_name = getattr(user, "display_name", getattr(user, "name", "User"))
    source_tag = "voice" if source.lower() == "voice" else "text"
    print(f"\n[{display_name} ({source_tag})]: {content}\n")

    if announce:
        await channel.send(f"🎙️ **{display_name}**: {content}")

    guild = user.guild if isinstance(user, discord.Member) else getattr(channel, "guild", None)
    guild_id = guild.id if guild else None

    async with channel.typing():
        loop = asyncio.get_running_loop()

        def _call_llm(prompt: str):
            return llm_json_sync(prompt)

        result = await loop.run_in_executor(
            None,
            persona_manager.step,
            str(user.id),
            guild_id,
            content,
            _call_llm,
        )
        print("[DEBUG llm_json]", result)


    response = str(result.get("reply", ""))
    mood = result.get("mood", "neutral")
    voice_cfg = result.get("voice") or {}
    vts_params = result.get("vts_params", {})
    voice_label = voice_cfg.get("voice_name") or voice_cfg.get("voice_id") or MELO_TTS_SPEAKER

    print(f"[Persona] mood={mood}, voice={voice_label} | reply={response[:80]}...")

    for part in chunk_message(response):
        await channel.send(part)

    ws = None
    if VTS_ENABLED:
        try:
            ws = await vts_connect_with_token()
            if ws:
                await vts_apply_params(ws, vts_params)
        except Exception as e:
            print(f"VTS 연결 실패(무시하고 진행): {e}")
            ws = None

    lock_key = guild.id if guild else 0
    lock = guild_locks[lock_key]
    async with lock:
        vc: Optional[discord.VoiceClient] = None
        if isinstance(user, discord.Member):
            vc = await get_or_connect_voice_client(user, channel)

        if vc:
            audio_payload = await synthesize_speech_async(response, voice_cfg)
            if audio_payload:
                audio_bytes, mime_type = audio_payload
                if ws:
                    est_duration = max(2.5, len(audio_bytes) / 32000)
                    asyncio.create_task(simulate_mouth(ws, est_duration))
                await play_audio_bytes_in_discord(vc, audio_bytes, mime_type, volume=VOICE_PLAYBACK_VOLUME)
                if isinstance(channel, discord.TextChannel):
                    await ensure_voice_listener(vc, channel)


async def play_audio_bytes_in_discord(vc: discord.VoiceClient, audio_bytes: bytes, mime_type: str = "audio/mpeg", volume: float = 1.0):
    if not audio_bytes:
        return

    ext = ".mp3"
    mt = (mime_type or "").lower()
    if "wav" in mt:
        ext = ".wav"
    elif "ogg" in mt:
        ext = ".ogg"
    elif "webm" in mt:
        ext = ".webm"

    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        await ensure_bot_voice_flags(vc)

        while vc.is_playing() or vc.is_paused():
            await asyncio.sleep(0.1)

        ffmpeg_options = '-vn -ac 2 -ar 48000'
        pcm_source = FFmpegPCMAudio(
            source=tmp_path,
            executable=FFMPEG_PATH,
            before_options="-nostdin",
            options=ffmpeg_options,
        )
        vol = max(0.0, float(volume))
        audio_source = discord.PCMVolumeTransformer(pcm_source, volume=vol)

        print("Starting audio playback (PCM)")
        vc.play(audio_source)

        while vc.is_playing():
            await asyncio.sleep(0.2)

        print("Audio playback finished")
    except FileNotFoundError:
        print("ffmpeg executable not found. Set FFMPEG_PATH in .env or install ffmpeg.")
    except Exception as e:
        print(f"Audio playback error: {e}")
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


async def get_or_connect_voice_client(member: discord.Member, channel: discord.abc.Messageable):
    if not member.voice or not member.voice.channel:
        try:
            await channel.send("Join a voice channel first!")
        except Exception:
            pass
        return None
    voice_channel = member.voice.channel
    vc = discord.utils.get(client.voice_clients, guild=member.guild)
    if vc and vc.is_connected():
        if vc.channel != voice_channel:
            await vc.move_to(voice_channel)
        return vc
    return await voice_channel.connect()


intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents, voice_receive=True)


@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name='AI VTuber (Persona Mode)'))
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return
    if str(message.channel.id) != str(DISCORD_CHANNEL_ID):
        return

    await handle_user_input(message.author, message.channel, message.content, source="text", announce=False)


if __name__ == "__main__":
    print("\nRunning Discord AI VTuber (Persona Mode) ...\n")
    client.run(DISCORD_TOKEN)
