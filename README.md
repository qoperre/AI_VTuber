# AI-Vtuber
This project reads Discord chat messages, replies with Google's Gemini language model, and now speaks by default through **MeloTTS** (Japanese neural voice) while listening with **OpenAI Whisper** (Korean speech-to-text). When the bot is in a voice channel it can hear participants and answer aloud in real time.


# Setup
Install dependencies:
```
git clone https://github.com/Koischizo/AI-Vtuber/
cd AI-Vtuber
python -m pip install -r requirements.txt
```

Additional requirements:
- [`ffmpeg`](https://ffmpeg.org/) (and `ffprobe`) available on your PATH for playback/transcoding.
- The first run will download MeloTTS and Whisper models. A CUDA-capable GPU is recommended but not required (set `MELO_TTS_DEVICE`/`WHISPER_DEVICE` to `cpu` for CPU-only).


# Usage
1. Copy `.env` and fill in the values:
   - `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID`, `GEMINI_KEY`
   - Optional tuning: `MELO_TTS_*`, `WHISPER_*`, `VOICE_LISTEN_*`, `VOICE_PLAYBACK_VOLUME`
2. (Optional) Adjust persona prompts or voice mapping in `persona/configs/rei.json`.
3. Run the bot:
   ```
   python run.py
   ```
4. Invite the bot to a text + voice channel, talk or type, and it will respond with Japanese speech synthesized by MeloTTS while transcribing Korean speech via Whisper.

# MeloTTS & Whisper Setup
1) Install dependencies: `python -m pip install -r requirements.txt`.
2) The first run downloads MeloTTS and Whisper models automatically. Set `MELO_TTS_DEVICE` / `WHISPER_DEVICE` to `cpu` if you do not have a GPU.
3) To switch MeloTTS speakers, change `MELO_TTS_SPEAKER` in `.env` or via the GUI.
4) Whisper defaults to Korean transcription (`WHISPER_LANGUAGE=ko`). Select a smaller model (e.g. `small`, `base`) if `medium` is too heavy.

# Troubleshooting
- MeloTTS not installed: ensure `pip install melotts` completed. The first inference downloads model weights; rerun the command if it was interrupted.
- Whisper GPU memory errors: switch to a smaller model via `WHISPER_MODEL` or set `WHISPER_DEVICE=cpu`.
- Audio playback issues: confirm `ffmpeg`/`ffprobe` are on PATH and the bot has permission to speak in the voice channel.

# Other
I used [This VTS plugin](https://lualucky.itch.io/vts-desktop-audio-plugin) and [VB Audio cable](https://vb-audio.com/Cable/) to make her mouth move and be able to play music at the same time

Please note that this project was created solely for fun and as part of a YouTube video, so the quality and reliability of the code may be questionable. Also, after the completion of the project checklist, there won't be much activity in updating or improving this repository. Nonetheless, we hope that this project can serve as a source of inspiration for anyone interested in building their own AI Vtuber.

- [x] Clean up
- [ ] GUI
- [ ] Executables (exe, bat or sh)
- [ ] Extra features (maybe) (Prompt injection protection, questions only mode, virtual audio)

# License
This program is under the [MIT license](/LICENSE) 

