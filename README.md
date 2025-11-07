# AI-Vtuber
This project reads Discord chat messages, replies with Google's Gemini language model, and now speaks by default through **Fish-Speech** (Japanese – Girl voice) while listening with **OpenAI Whisper** (Korean speech-to-text). When the bot is in a voice channel it can hear participants and answer aloud in real time.


# Setup
Install dependencies:
```
git clone https://github.com/Koischizo/AI-Vtuber/
cd AI-Vtuber
python -m pip install -r requirements.txt
```

Additional requirements:
- [`ffmpeg`](https://ffmpeg.org/) (and `ffprobe`) available on your PATH for playback/transcoding (Fish-Speech uses it when preparing reference audio).
- Download the open-source Fish-Speech checkpoints: `python -m fish_speech.tools.download_models`. This pulls the **OpenAudio S1-mini** model into `checkpoints/openaudio-s1-mini/` (see `.env`).
- Prepare a voice reference for the Japanese girl persona by placing one or more clean `.wav` files (and optional `.lab` transcripts) under `references/japanese_girl/`. The default persona configuration points to this folder.
- Whisper still benefits from a CUDA-capable GPU, but it can run on CPU (`WHISPER_DEVICE=cpu`).


# Usage
1. Copy `.env` and fill in the values:
   - `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID`, `GEMINI_KEY`
   - Fish-Speech: `FISH_SPEECH_CHECKPOINT_DIR`, `FISH_SPEECH_DECODER_CHECKPOINT`, `FISH_SPEECH_REFERENCE_DIR`, and other optional tuning knobs (`FISH_SPEECH_*`)
   - Whisper / audio options: `WHISPER_*`, `VOICE_LISTEN_*`, `VOICE_PLAYBACK_VOLUME`
2. (Optional) Adjust persona prompts or voice mapping in `persona/configs/rei.json`.
3. Run the bot:
   ```
   python run.py
   ```
4. Invite the bot to a text + voice channel, talk or type, and it will respond with Japanese speech synthesized by Fish-Speech while transcribing Korean speech via Whisper.

# Fish-Speech & Whisper Setup
1) Install dependencies: `python -m pip install -r requirements.txt`.
2) Run `python -m fish_speech.tools.download_models` to pull the OpenAudio S1-mini checkpoints into `checkpoints/openaudio-s1-mini/`. Update `.env` if you store them elsewhere.
3) Provide reference audio for the Japanese girl voice. Drop one or more short voice clips into `references/japanese_girl/` (the filename `.lab` counterpart can contain the spoken text to improve conditioning). Adjust `FISH_SPEECH_REFERENCE_DIR` in `.env`/persona JSON if you use a different folder.
4) Whisper downloads its model on first use. Set `WHISPER_DEVICE=cpu` if you do not have a GPU, and consider switching to a smaller model (`small`, `base`) if memory is tight.

# Troubleshooting
- Fish-Speech load failures: confirm the checkpoints exist at `FISH_SPEECH_CHECKPOINT_DIR` and `FISH_SPEECH_DECODER_CHECKPOINT`. Re-run `python -m fish_speech.tools.download_models` if files are missing.
- Voice quality issues: make sure `references/japanese_girl/` contains clean recordings that match the intended style. Each `.wav` may optionally have a `.lab` transcript file with the spoken line.
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

