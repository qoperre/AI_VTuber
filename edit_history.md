# Edit History

- **2025-10-18**: Migrated the application from using YouTube chat as an input source to using a Discord bot.
    - Replaced `pytchat` with `discord.py`.
    - Updated `run.py` to connect to Discord and handle messages.
    - Modified `config.json` to include Discord bot token and channel ID.
    - Updated `README.md` with new setup and usage instructions.

- **2025-10-19**:
    - Modified `run.py` to address message length and latency issues.
    - Changed `llm` function to be asynchronous using `async def` and `await model.generate_content_async()`.
    - Updated `on_message` to `await llm()` and handle responses longer than 2000 characters by splitting them into multiple messages.
    - Moved the synchronous `tts_play` function to a separate thread using `asyncio.to_thread` to prevent blocking.