# Explain: Functions and Classes

This document summarizes key functions, classes, and their responsibilities across the project. It focuses on behavior, inputs, outputs, and noteworthy side effects to help you quickly navigate and extend the codebase.


## File: run.py

- def _extract_first_json(text: str) -> dict|None
  - Purpose: Extract the first valid top-level JSON object from a free-form model response, including fenced code blocks (```json ... ```).
  - Params: text (raw string from LLM)
  - Returns: Parsed dict or None if not found.

- def llm_json_sync(prompt: str) -> dict
  - Purpose: Calls Gemini synchronously and parses the response into a small JSON object ({reply, mood_delta, affinity_delta}).
  - Params: prompt (constructed by PersonaManager)
  - Returns: dict with sane defaults if parsing fails.

- async def vts_connect_with_token(retries=1, timeout_sec=2.0) -> websockets.WebSocketClientProtocol|None
  - Purpose: Connect to VTube Studio WebSocket API with stored token; returns a live socket or None.
  - Notes: Reads `vts_token.json` and sends authentication. Retries limited.

- async def vts_set_parameter(ws, name: str, value: float) -> None
  - Purpose: Sends a parameter update to VTS (e.g., MouthOpen/Smile).

- async def vts_apply_params(ws, params: dict) -> None
  - Purpose: Convenience loop to set a batch of VTS parameters.

- async def simulate_mouth(ws, duration=3.5) -> None
  - Purpose: Temporarily animates mouth parameters while audio is playing.

- async def ensure_bot_voice_flags(vc: discord.VoiceClient) -> None
  - Purpose: Makes sure the voice client is in a good state for playback (nop guard in current form).

- def _normalize_voice_config(voice_cfg: dict) -> dict
  - Purpose: Normalize persona voice data into Fish-Speech offline parameters (reference folder/files, sampling knobs).
- def _resolve_device(preferred: str) -> str
  - Purpose: Decide between GPU/CPU automatically for Whisper model loading.
- def _collect_fish_references(info: dict) -> list[ServeReferenceAudio]
  - Purpose: Load reference audio/text pairs from persona config (`reference_files` or `reference_dir`).
- async def _ensure_fish_engine() -> TTSInferenceEngine
  - Purpose: Lazily download/load OpenAudio checkpoints and build the Fish-Speech inference engine on the configured device.
- def _render_wave(audio: np.ndarray, sample_rate: int) -> bytes
  - Purpose: Convert float PCM into a WAV byte stream for Discord playback.
- async def fish_speech_async(text: str, voice_cfg: dict) -> tuple[bytes, str]|None
  - Purpose: Generate speech locally via the Fish-Speech inference engine and return raw WAV bytes plus mime type.
- async def synthesize_speech_async(message: str, voice_cfg: dict|None) -> tuple[bytes, str]|None
  - Purpose: High-level TTS entrypoint (Fish-Speech-first; other providers could be added later).
- async def _ensure_whisper_model()
  - Purpose: Lazy-load the Whisper model according to .env configuration.
- async def transcribe_audio_async(audio_bytes: bytes, *, sample_rate=48000, language_code: str|None=None, audio_channel_count: int = 2) -> str
  - Purpose: Transcribe captured PCM audio using Whisper and return normalized text.
- async def process_voice_sink(sink, vc: discord.VoiceClient, text_channel: discord.TextChannel) -> None
  - Purpose: Extract per-user PCM from `discord.sinks` WaveSink, call STT, and dispatch recognized text back into the same pipeline as text chat.
  - Notes: Skips the bot’s own audio. Truncates overly long transcripts.

- async def voice_listener_loop(guild_id: int, vc: discord.VoiceClient) -> None
  - Purpose: Periodically starts a WaveSink recording window, waits, stops, then processes chunk with STT.
  - Notes: Controlled by `VOICE_LISTEN_ENABLED` and presence of `discord.sinks` in your environment.

- async def ensure_voice_listener(vc: discord.VoiceClient, text_channel: discord.TextChannel) -> None
  - Purpose: Starts or keeps the guild-specific listener task running and ties it to a text channel for responses.

- async def handle_user_input(user, channel, content: str, *, source="text", announce=False) -> None
  - Purpose: Core orchestration per user turn. Records history, builds prompt via Persona, calls LLM, posts text reply, synthesizes TTS, plays it in voice, and ensures listener.
  - Side effects: VTS connect/apply, playback to Discord, sends messages, updates memory.

- async def play_audio_bytes_in_discord(vc: discord.VoiceClient, audio_bytes: bytes, mime_type="audio/mpeg", volume=1.0) -> None
  - Purpose: Writes audio to a temp file and streams via FFmpegOpusAudio to the voice channel.
  - Notes: Respects `FFMPEG_PATH`; cleans up temp file.

- def chunk_message(text: str, limit=1900) -> list[str]
  - Purpose: Split long messages into Discord-safe chunks preserving newlines.

- async def get_or_connect_voice_client(member: discord.Member, channel) -> discord.VoiceClient|None
  - Purpose: Join or move to member’s current voice channel and return a ready voice client.

- async def on_ready() -> None
  - Purpose: Discord event hook. Sets presence and logs bot identity.

- async def on_message(message: discord.Message) -> None
  - Purpose: Discord event hook. Routes matching channel messages into `handle_user_input`.


## File: persona/persona_manager.py

- class PersonaManager
  - __init__(rei_config_path: str, memory_store: MemoryStore, base_dir: str|None)
    - Loads persona config (rei.json), wires EmotionEngine and MemoryStore.
  - _maybe_learn_name(k: MemoryKey, user_text: str) -> None
    - Heuristically extracts and remembers a user-provided name in profile storage.
  - select_voice(mood: str) -> dict
    - Picks voice config for a mood from rei.json voice_map.
  - vts_params(mood: str) -> dict
    - Maps mood to VTS parameter set from rei.json.
  - _known_facts_line(k: MemoryKey) -> str
    - Collapses known profile facts (e.g., name) into a hint line for prompts.
  - _recent_context_block(k: MemoryKey) -> str
    - Returns recent conversation turns or a compact summary if too long.
  - build_prompt(k: MemoryKey, user_text: str) -> str
    - Constructs the LLM prompt including mood, style hints, facts, and recent context.
  - step(user_id: str, guild_id: int|None, user_text: str, llm_call) -> dict
    - Orchestrates a single user step: save history, call LLM, update emotion and affinity, return reply + voice/VTS info.


## File: persona/emotion_engine.py

- @dataclass EmotionState
  - Fields: mood, intensity, valence, arousal, last_update_ts.

- class EmotionEngine
  - __init__(config: dict)
    - Reads engine tuning, compiles sentiment regexes, sets decay and biases.
  - _compile_keywords(kw_cfg: dict[str, list[str]]) -> dict
    - Compiles exact-match regex patterns for fast keyword checks.
  - _score_sentiment(text: str) -> tuple[float, float]
    - Heuristic valence and arousal deltas based on keyword categories.
  - _apply_decay(st: EmotionState, dt: float) -> None
    - Applies time-based decay to intensity/arousal and relaxes valence toward 0.
  - _pick_mood_from_pad(v: float, a: float, hint: str|None) -> str
    - Maps PAD-like coordinates to a discrete mood, optionally honoring a forced transition.
  - _apply_transition_triggers(text: str) -> str|None
    - Returns a target mood if any transition trigger word is present.
  - update(user_text: str, llm_signal: dict, affinity: int) -> EmotionState
    - Full state update: decay, user sentiment, LLM signal, affinity bias, transition, cooldown filtering.


## File: persona/memory_store.py

- @dataclass(frozen=True) MemoryKey
  - Fields: platform, scope, user_id

- def make_key_discord(user_id: str, guild_id: int|None) -> MemoryKey
  - Purpose: Build a normalized key for Discord DM vs guild scope.

- class MemoryStore
  - __init__(base_dir: str, stm_maxlen: int=10)
    - Prepares base folders and per-root locks; controls STM maximum window length.
  - get_profile(k: MemoryKey) -> dict
  - set_profile_field(k: MemoryKey, field: str, value, source="system", confidence=0.95, ttl_sec: float|None=None) -> None
  - get_profile_value(k: MemoryKey, field: str) -> Any|None
  - remember_name(k: MemoryKey, name: str, source="user_statement") -> None
  - get_name(k: MemoryKey) -> str|None
  - add_history(k: MemoryKey, role: str, text: str) -> None
  - get_history(k: MemoryKey, n: int=6) -> list[tuple[str, str]]
  - get_summary(k: MemoryKey) -> dict
  - set_summary(k: MemoryKey, text: str, covered_turns: int) -> None
  - get_affinity(k: MemoryKey) -> int
  - add_affinity(k: MemoryKey, delta: int) -> int
  - decay_affinity(k: MemoryKey, toward=0.0, rate_per_sec=0.001) -> None
  - clear_user(k: MemoryKey) -> None
    - Notes: JSON files stored under `mem/<platform>/<scope>/<user_id>/`.


## File: persona/summarizer.py

- def simple_dialogue_summary(turns: list[tuple[str, str]], max_chars=180) -> str
  - Purpose: Compactly summarize the last few turns, truncating per-turn and overall length.


## File: get_vtubestudio_token.py

- async def main() -> None
  - Purpose: Connect to VTS WebSocket, request an authentication token for this plugin, and write `vts_token.json`.
  - Usage: `python get_vtubestudio_token.py` while VTS is running.


## File: run_gui.py

- def run_process(voice_mode: str, line_callback: callable|None=None) -> None
  - Purpose: Spawn `python run.py` in a background thread with environment variables set for the selected voice path.
  - Notes: Streams stdout to an optional UI callback.

- def stop_process() -> None
  - Purpose: Stop the background process cleanly.

- class RunGUI
  - __init__(): Builds a small Tkinter UI (voice selection, console, run/stop buttons).
  - run(): Starts the Tkinter mainloop.


## File: discordbot-test/Test-AI.py (example bot)

- async def on_ready()
  - Purpose: Show basic login info and set presence for the sample bot.

- @client.command() async def hello(ctx)
  - Purpose: Responds with a simple greeting (demo).

- @client.command() async def login(ctx)
  - Purpose: Sends a formatted message with the caller identity (demo).


## Notes

- JSON configs (e.g., `persona/configs/rei.json`, `mem/.../stm.json`, `mem/.../affinity.json`) do not define functions, but their structure is consumed directly by the modules above.
- Environment variables in `.env` drive model selection, Google credentials, Discord token, and voice listener toggles.

