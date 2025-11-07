import os
import sys
import re
import json
import asyncio
import tempfile
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from pathlib import Path
from discord import FFmpegPCMAudio
import discord.abc

from dotenv import load_dotenv
import discord
import websockets
try:
    import google.generativeai as genai  # lazy-configured below
except Exception:
    genai = None  # optional
import time
import wave
import numpy as np
import httpx
import ormsgpack
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore
# Defer fish-speech imports to runtime when engine mode is used, to avoid protobuf conflicts.
TTSInferenceEngine = None  # type: ignore
ServeReferenceAudio = None  # type: ignore
ServeTTSRequest = None  # type: ignore
audio_to_bytes = None  # type: ignore
fish_load_decoder_model = None  # type: ignore
fish_launch_queue = None  # type: ignore
AUDIO_EXTENSIONS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]

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
DEBUG_SAVE_TTS_WAV        = os.getenv("DEBUG_SAVE_TTS_WAV", "false").lower() == "true"
DEBUG_SAVE_TTS_WAV_PATH   = os.getenv("DEBUG_SAVE_TTS_WAV_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.wav"))
DEBUG_TTS_LOGS           = os.getenv("DEBUG_TTS_LOGS", "false").lower() == "true"

# Fish-Speech (Japanese TTS) + Whisper STT defaults
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "FISH_SPEECH").upper()
FISH_SPEECH_MODE = os.getenv("FISH_SPEECH_MODE", "server").strip().lower()
FISH_SPEECH_SERVER_URL = os.getenv("FISH_SPEECH_SERVER_URL", "http://127.0.0.1:8080/v1/tts").strip()


def _parse_int_env(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

def _parse_float_env(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _parse_bool_env(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


FISH_SPEECH_CHECKPOINT_DIR = os.getenv(
    "FISH_SPEECH_CHECKPOINT_DIR",
    os.path.join(BASE_DIR, "checkpoints", "openaudio-s1-mini"),
)
FISH_SPEECH_DECODER_CHECKPOINT = os.getenv(
    "FISH_SPEECH_DECODER_CHECKPOINT",
    os.path.join(FISH_SPEECH_CHECKPOINT_DIR, "codec.pth"),
)
FISH_SPEECH_DECODER_CONFIG = os.getenv(
    "FISH_SPEECH_DECODER_CONFIG", "modded_dac_vq"
)
FISH_SPEECH_REFERENCE_ID = os.getenv("FISH_SPEECH_REFERENCE_ID", "japanese_girl")
FISH_SPEECH_REFERENCE_DIR = os.getenv(
    "FISH_SPEECH_REFERENCE_DIR",
    os.path.join(BASE_DIR, "references", FISH_SPEECH_REFERENCE_ID),
)
FISH_SPEECH_MAX_NEW_TOKENS = _parse_int_env(
    os.getenv("FISH_SPEECH_MAX_NEW_TOKENS"), 1024
)
FISH_SPEECH_CHUNK_LENGTH = _parse_int_env(
    os.getenv("FISH_SPEECH_CHUNK_LENGTH"), 300
)
FISH_SPEECH_TOP_P = _parse_float_env(os.getenv("FISH_SPEECH_TOP_P"), 0.8)
FISH_SPEECH_TEMPERATURE = _parse_float_env(
    os.getenv("FISH_SPEECH_TEMPERATURE"), 0.8
)
FISH_SPEECH_HTTP_TIMEOUT = _parse_float_env(os.getenv("FISH_SPEECH_HTTP_TIMEOUT"), 900.0)
FISH_SPEECH_REPETITION_PENALTY = _parse_float_env(
    os.getenv("FISH_SPEECH_REPETITION_PENALTY"), 1.1
)
FISH_SPEECH_USE_MEMORY_CACHE = (
    "on" if _parse_bool_env(os.getenv("FISH_SPEECH_USE_MEMORY_CACHE", "on"), True) else "off"
)
FISH_SPEECH_SEED = os.getenv("FISH_SPEECH_SEED")
FISH_SPEECH_DEVICE = os.getenv("FISH_SPEECH_DEVICE", "auto")
FISH_SPEECH_PRECISION = os.getenv("FISH_SPEECH_PRECISION", "bf16").lower()
FISH_SPEECH_COMPILE = _parse_bool_env(os.getenv("FISH_SPEECH_COMPILE"), False)
FISH_SPEECH_WARMUP = _parse_bool_env(os.getenv("FISH_SPEECH_WARMUP"), False)

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
MODEL = None

def _ensure_genai_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if genai is None:
        raise RuntimeError("google-generativeai not installed; set GEMINI_KEY or install google-generativeai")
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_KEY not provided")
    genai.configure(api_key=GEMINI_KEY)
    MODEL = genai.GenerativeModel(GEMINI_MODEL)
    return MODEL

_whisper_model: Optional[Any] = None
_whisper_model_lock = asyncio.Lock()
_fish_speech_engine: Optional[TTSInferenceEngine] = None
_fish_speech_lock = asyncio.Lock()

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
    # Ensure Gemini model is initialized lazily. If unavailable, fall back safely.
    try:
        model = _ensure_genai_model()
    except Exception as e:
        print(f"Gemini init failed in llm_json_sync: {e}")
        return {"reply": "了解だよ！", "mood_delta": 0.0, "affinity_delta": 0}

    resp = model.generate_content(prompt + "\n\n" + guard)
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
    cfg = dict(voice_cfg or {})
    provider = str(cfg.get("provider") or TTS_PROVIDER).upper()

    def _to_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(fallback)

    def _to_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    reference_id = str(
        cfg.get("reference_id")
        or cfg.get("voice_id")
        or cfg.get("voice_name")
        or FISH_SPEECH_REFERENCE_ID
        or ""
    ).strip()

    ref_dir_cfg = cfg.get("reference_directory") or cfg.get("reference_dir")
    reference_dir = (
        str(ref_dir_cfg).strip()
        if ref_dir_cfg
        else (FISH_SPEECH_REFERENCE_DIR or "")
    )
    if reference_dir:
        reference_dir = os.path.abspath(
            reference_dir
            if os.path.isabs(reference_dir)
            else os.path.join(BASE_DIR, reference_dir)
        )

    reference_files_cfg = cfg.get("reference_files") or cfg.get("references") or []
    if isinstance(reference_files_cfg, (str, os.PathLike)):
        reference_files = [str(reference_files_cfg)]
    elif isinstance(reference_files_cfg, (list, tuple)):
        reference_files = [str(p) for p in reference_files_cfg if p]
    else:
        reference_files = []
    reference_files = [
        os.path.abspath(p) if not os.path.isabs(p) else p for p in reference_files
    ]

    audio_format = str(
        cfg.get("format") or cfg.get("audio_format") or "wav"
    ).lower()
    use_memory_cache = str(
        cfg.get("use_memory_cache") or FISH_SPEECH_USE_MEMORY_CACHE or "off"
    ).lower()

    max_new_tokens = _to_int(cfg.get("max_new_tokens") or FISH_SPEECH_MAX_NEW_TOKENS, 1024)
    chunk_length = _to_int(cfg.get("chunk_length") or FISH_SPEECH_CHUNK_LENGTH, 300)
    top_p = _to_float(cfg.get("top_p") or FISH_SPEECH_TOP_P, 0.8)
    temperature = _to_float(cfg.get("temperature") or FISH_SPEECH_TEMPERATURE, 0.8)
    repetition_penalty = _to_float(
        cfg.get("repetition_penalty") or FISH_SPEECH_REPETITION_PENALTY,
        1.1,
    )
    seed = cfg.get("seed", FISH_SPEECH_SEED)
    if seed in ("", None):
        seed = None
    else:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    return {
        "provider": provider,
        "reference_id": reference_id,
        "reference_dir": reference_dir,
        "reference_files": reference_files,
        "format": audio_format,
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "use_memory_cache": use_memory_cache,
        "seed": seed,
        "voice_name": cfg.get("voice_name") or cfg.get("voice_id") or "",
        "raw": cfg,
    }


def _resolve_fish_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def _resolve_fish_device(preferred: str) -> str:
    pref = (preferred or "").strip().lower()
    if pref in ("", "auto"):
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"
    return preferred


def _fish_precision_dtype():
    if torch is None:
        return None
    prec = (FISH_SPEECH_PRECISION or "").lower()
    if prec in ("fp16", "float16", "half"):
        return torch.float16
    if prec in ("fp32", "float32", "full"):
        return torch.float32
    return torch.bfloat16


def _collect_fish_references(info: Dict[str, Any]) -> List[ServeReferenceAudio]:
    references: List[ServeReferenceAudio] = []
    if ServeReferenceAudio is None:
        return references

    def _load_reference(path_str: str) -> Optional[ServeReferenceAudio]:
        try:
            audio_bytes = audio_to_bytes(path_str)
        except Exception as e:
            print(f"Fish-Speech reference load failed ({path_str}): {e}")
            return None
        lab_path = os.path.splitext(path_str)[0] + ".lab"
        text_hint = ""
        if os.path.exists(lab_path):
            try:
                with open(lab_path, "r", encoding="utf-8") as lab_file:
                    text_hint = lab_file.read().strip()
            except Exception:
                text_hint = ""
        return ServeReferenceAudio(audio=audio_bytes, text=text_hint)

    for path in info.get("reference_files", []):
        ref = _load_reference(path)
        if ref:
            references.append(ref)

    if not references and info.get("reference_dir"):
        ref_dir = Path(info["reference_dir"])
        if ref_dir.is_dir():
            audio_exts = {ext.lower() for ext in AUDIO_EXTENSIONS}
            for file_path in sorted(ref_dir.iterdir()):
                if file_path.suffix.lower() in audio_exts:
                    ref = _load_reference(str(file_path))
                    if ref:
                        references.append(ref)
    return references


def _convert_to_numpy_audio(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    try:
        return np.asarray(data, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def _render_wave(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio is None or audio.size == 0:
        return b""
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(pcm16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


async def _ensure_fish_engine() -> TTSInferenceEngine:
    # Lazy import here to avoid protobuf conflicts in Gemini env
    global TTSInferenceEngine, ServeReferenceAudio, ServeTTSRequest, audio_to_bytes, fish_load_decoder_model, fish_launch_queue
    if torch is None:
        raise RuntimeError("PyTorch is required for fish-speech. Install torch>=2.1.0.")
    if TTSInferenceEngine is None:
        try:
            from fish_speech.inference_engine import TTSInferenceEngine as _TTSInferenceEngine  # type: ignore
            from fish_speech.models.dac.inference import (  # type: ignore
                load_model as _fish_load_decoder_model,
            )
            from fish_speech.models.text2semantic.inference import (  # type: ignore
                launch_thread_safe_queue as _fish_launch_queue,
            )
            from fish_speech.utils.file import audio_to_bytes as _audio_to_bytes, AUDIO_EXTENSIONS as _AUDIO_EXT  # type: ignore
            from fish_speech.utils.schema import (  # type: ignore
                ServeReferenceAudio as _ServeReferenceAudio,
                ServeTTSRequest as _ServeTTSRequest,
            )
            TTSInferenceEngine = _TTSInferenceEngine
            ServeReferenceAudio = _ServeReferenceAudio
            ServeTTSRequest = _ServeTTSRequest
            audio_to_bytes = _audio_to_bytes
            fish_load_decoder_model = _fish_load_decoder_model
            fish_launch_queue = _fish_launch_queue
            if _AUDIO_EXT:
                # Prefer library-provided list
                try:
                    from fish_speech.utils.file import AUDIO_EXTENSIONS as __list  # type: ignore
                    globals()["AUDIO_EXTENSIONS"] = list(__list)
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"fish-speech library not installed or failed to import: {e}")

    global _fish_speech_engine
    if _fish_speech_engine is not None:
        return _fish_speech_engine

    async with _fish_speech_lock:
        if _fish_speech_engine is not None:
            return _fish_speech_engine

        def _load_engine() -> TTSInferenceEngine:
            checkpoint_dir = _resolve_fish_path(FISH_SPEECH_CHECKPOINT_DIR)
            decoder_checkpoint = _resolve_fish_path(FISH_SPEECH_DECODER_CHECKPOINT)
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(
                    f"Fish-Speech checkpoint directory not found: {checkpoint_dir}"
                )
            if not os.path.isfile(decoder_checkpoint):
                raise FileNotFoundError(
                    f"Fish-Speech decoder checkpoint not found: {decoder_checkpoint}"
                )

            device = _resolve_fish_device(FISH_SPEECH_DEVICE)
            precision = _fish_precision_dtype() or (
                torch.float16 if device != "cpu" else torch.float32
            )

            llama_queue = fish_launch_queue(
                checkpoint_path=checkpoint_dir,
                device=device,
                precision=precision,
                compile=FISH_SPEECH_COMPILE,
            )
            decoder_model = fish_load_decoder_model(
                config_name=FISH_SPEECH_DECODER_CONFIG,
                checkpoint_path=decoder_checkpoint,
                device=device,
            )

            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=FISH_SPEECH_COMPILE,
            )

            if FISH_SPEECH_WARMUP:
                try:
                    warmup_request = ServeTTSRequest(
                        text="こんにちは、よろしくお願いします。",
                        references=[],
                        reference_id=None,
                        format="wav",
                        max_new_tokens=256,
                        chunk_length=200,
                        top_p=0.8,
                        repetition_penalty=1.05,
                        temperature=0.7,
                        streaming=False,
                        use_memory_cache="on",
                        normalize=True,
                    )
                    for _ in engine.inference(warmup_request):
                        pass
                except Exception as warmup_error:
                    print(f"Fish-Speech warmup failed (ignored): {warmup_error}")

            return engine

        loop = asyncio.get_running_loop()
        _fish_speech_engine = await loop.run_in_executor(None, _load_engine)
        return _fish_speech_engine


def _resolve_device(preferred: str) -> str:
    pref = (preferred or "").lower()
    if pref in ("", "auto"):
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return preferred


async def fish_speech_async(text: str, voice_cfg: Dict[str, Any]) -> Optional[Tuple[bytes, str]]:
    info = _normalize_voice_config(voice_cfg)

    # Server mode: call local Fish-Speech API (tools.api_server)
    if FISH_SPEECH_MODE == "server":
        try:
            payload: Dict[str, Any] = {
                "text": str(text)[:5000],
                "references": [],
                "reference_id": info.get("reference_id") or None,
                "format": "wav",
                "max_new_tokens": int(max(0, info.get("max_new_tokens", 0))),
                "chunk_length": int(max(100, min(300, info.get("chunk_length", 200)))),
                "top_p": float(info.get("top_p", 0.8)),
                "repetition_penalty": float(info.get("repetition_penalty", 1.1)),
                "temperature": float(info.get("temperature", 0.8)),
                "streaming": False,
                "use_memory_cache": "on" if info.get("use_memory_cache") == "on" else "off",
                "normalize": True,
                "seed": info.get("seed"),
            }
            timeout = float(FISH_SPEECH_HTTP_TIMEOUT or 30.0)
            if DEBUG_TTS_LOGS:
                print(f"[DEBUG] Fish-Speech request (server) -> {FISH_SPEECH_SERVER_URL} | text_len={len(payload['text'])} | timeout={timeout}")
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    FISH_SPEECH_SERVER_URL,
                    content=ormsgpack.packb(payload),
                    headers={"content-type": "application/msgpack"},
                )
            if DEBUG_TTS_LOGS:
                length = len(resp.content) if resp.content else 0
                print(f"[DEBUG] Fish-Speech response status={resp.status_code} bytes={length}")
            if resp.status_code != 200:
                if DEBUG_TTS_LOGS:
                    print(f"[DEBUG] Fish-Speech response body: {resp.text[:200]}")
                print(f"Fish-Speech server error {resp.status_code}: {resp.text[:200]}")
                return None
            audio_bytes = resp.content
            if not audio_bytes:
                print("Fish-Speech server returned empty audio.")
                return None
            return audio_bytes, "audio/wav"
        except Exception as e:
            if DEBUG_TTS_LOGS:
                print(f"[DEBUG] Fish-Speech server request exception: {e}")
            print(f"Fish-Speech server request failed: {e}")
            return None

    # Engine mode: run inference in-process (requires fish_speech + protobuf<3.20)
    try:
        engine = await _ensure_fish_engine()
    except Exception as e:
        print(f"Fish-Speech init failed: {e}")
        return None

    references = _collect_fish_references(info)
    reference_id = None if references else (info.get("reference_id") or None)

    chunk_length = int(max(100, min(300, info.get("chunk_length", 200))))
    max_new_tokens = int(max(0, info.get("max_new_tokens", 0)))
    top_p = float(info.get("top_p", 0.8))
    temperature = float(info.get("temperature", 0.8))
    repetition_penalty = float(info.get("repetition_penalty", 1.1))
    use_memory_cache = "on" if info.get("use_memory_cache") == "on" else "off"
    seed = info.get("seed")

    def _generate() -> Optional[Tuple[int, np.ndarray]]:
        request = ServeTTSRequest(
            text=str(text)[:5000],
            references=references,
            reference_id=reference_id,
            format="wav",
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            streaming=False,
            use_memory_cache=use_memory_cache,
            normalize=True,
            seed=seed,
        )
        sample_rate: Optional[int] = None
        collected: List[np.ndarray] = []
        for result in engine.inference(request):
            code = getattr(result, "code", "")
            if code == "error":
                err = getattr(result, "error", None)
                if isinstance(err, Exception):
                    raise err
                raise RuntimeError(str(err) if err else "Fish-Speech inference error")
            audio_tuple = getattr(result, "audio", None)
            if not (isinstance(audio_tuple, tuple) and len(audio_tuple) >= 2):
                continue
            sr = int(audio_tuple[0])
            audio_np = _convert_to_numpy_audio(audio_tuple[1])
            if audio_np.size == 0:
                continue
            sample_rate = sr or sample_rate
            collected.append(audio_np)
            if code == "final":
                break
        if not collected:
            return None
        final_audio = np.concatenate(collected, axis=0) if len(collected) > 1 else collected[0]
        return (sample_rate or 44100, final_audio)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _generate)
    except Exception as e:
        print(f"Fish-Speech synthesis failed: {e}")
        return None
    if not result:
        print("Fish-Speech returned empty audio.")
        return None
    sample_rate, audio_np = result
    audio_bytes = _render_wave(audio_np, sample_rate)
    return (audio_bytes, "audio/wav") if audio_bytes else None
async def synthesize_speech_async(message: str, voice_cfg: Optional[Dict[str, Any]] = None) -> Optional[Tuple[bytes, str]]:
    text = (message or "").strip()
    if not text:
        return None
    voice_cfg = voice_cfg or {}
    provider = str(voice_cfg.get("provider") or TTS_PROVIDER).upper()

    if provider in ("FISH_SPEECH", "FISH", "FISH-SPEECH"):
        audio = await fish_speech_async(text, voice_cfg)
        if audio:
            if DEBUG_SAVE_TTS_WAV:
                try:
                    with open(DEBUG_SAVE_TTS_WAV_PATH, "wb") as f:
                        f.write(audio[0])
                    print(f"[DEBUG] Saved TTS WAV: {DEBUG_SAVE_TTS_WAV_PATH} ({len(audio[0])} bytes)")
                except Exception as e:
                    print(f"[DEBUG] Save TTS WAV failed: {e}")
            return audio
    elif provider == "GEMINI":
        try:
            _ensure_genai_model()
        except Exception as e:
            print(f"Gemini init failed: {e}; falling back to Fish-Speech.")
        audio = await fish_speech_async(text, voice_cfg)
        if audio:
            if DEBUG_SAVE_TTS_WAV:
                try:
                    with open(DEBUG_SAVE_TTS_WAV_PATH, "wb") as f:
                        f.write(audio[0])
                    print(f"[DEBUG] Saved TTS WAV: {DEBUG_SAVE_TTS_WAV_PATH} ({len(audio[0])} bytes)")
                except Exception as e:
                    print(f"[DEBUG] Save TTS WAV failed: {e}")
            return audio
    else:
        audio = await fish_speech_async(text, voice_cfg)
        if audio:
            if DEBUG_SAVE_TTS_WAV:
                try:
                    with open(DEBUG_SAVE_TTS_WAV_PATH, "wb") as f:
                        f.write(audio[0])
                    print(f"[DEBUG] Saved TTS WAV: {DEBUG_SAVE_TTS_WAV_PATH} ({len(audio[0])} bytes)")
                except Exception as e:
                    print(f"[DEBUG] Save TTS WAV failed: {e}")
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
    voice_label = (
        voice_cfg.get("voice_name")
        or voice_cfg.get("voice_id")
        or voice_cfg.get("reference_id")
        or "Fish-Speech"
    )

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
        print("[DEBUG] No audio bytes to play.")
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

        print(f"Starting audio playback (PCM): {tmp_path} | mime={mime_type} | volume={vol}")
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
