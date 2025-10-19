# AI-Vtuber
This code is designed to read chat messages from Discord and then utilize Google's Gemini language model to generate responses. The output from Gemini is then read out loud using a TTS (Text-to-Speech) engine provided by ElevenLabs.



# Setup
Install dependencies
```
git clone https://github.com/Koischizo/AI-Vtuber/
cd AI-Vtuber
pip install -r requirements.txt
```
It also requires [`ffmpeg`](https://ffmpeg.org/) to be installed

# Usage

Edit the variables `EL_key`, `GEMINI_key`, `discord_token`, and `discord_channel_id` in `config.json`

`EL_key` is the API key for [ElevenLabs](https://beta.elevenlabs.io/). Found in Profile Settings

`GEMINI_key` is the API key for Google's Gemini. Found [here](https://aistudio.google.com/app/apikey)

`discord_token` is your Discord bot's token.

`discord_channel_id` is the ID of the channel you want the bot to read messages from.

Then run `run.py`
```
python run.py
```
then you're set


# Other
I used [This VTS plugin](https://lualucky.itch.io/vts-desktop-audio-plugin) and [VB Audio cable](https://vb-audio.com/Cable/) to make her mouth move and be able to play music at the same time

Please note that this project was created solely for fun and as part of a YouTube video, so the quality and reliability of the code may be questionable. Also, after the completion of the project checklist, there won't be much activity in updating or improving this repository. Nonetheless, we hope that this project can serve as a source of inspiration for anyone interested in building their own AI Vtuber.

- [x] Clean up
- [ ] GUI
- [ ] Executables (exe, bat or sh)
- [ ] Extra features (maybe) (Prompt injection protection, questions only mode, virtual audio)

# License
This program is under the [MIT license](/LICENSE) 

