import discord
from discord.ext import commands

# client = discord.Client()

# 사용자가 명령어 입력 시 ./를 입력하고 명령어 입력
# client = commands.Bot(command_prefix='./')

# 신버전에서는 intents 인자 필수
# ================== 신버전 설정 =====================

# Intents 설정
intents = discord.Intents.default()
intents.message_content = True # 메세지 내용을 읽을 수 있도록

# Bot 생섬 시 intents를 전달해야 함
client = commands.Bot(command_prefix='./', intents=intents)

# ===================================================

# on_ready는 시작할 때 한 번만 실행
@client.event
async def on_ready():
    print('Login...')
    print(f'{client.user}에 로그인하셨습니다.')
    print(f'ID: {client.user.name}')
    await client.change_presence(status=discord.Status.online, activity=discord.Game('VS Code로 개발'))
    # Status 대문자, online: 온라인, idle: 자리 비움, dnd: 방해 금지

# @client.event
# async def on_message(message):
#     if message.author == client.user:
#         return  # 봇 자신의 메시지 무시

#     # message.content.startswith()는 해당 문자로 시작하는 단어에 대해서
#     # 인식하여 메세지 전송, ==로 비교 시 해당 문자만 인식
#     if message.content.startswith('테스트'):
#         await message.channel.send("{} | {}, 안녕!".format(message.author, message.author.mention))

#     if message.content == '테스트':
#         # 채널에 메세지 전송
#         await message.channel.send("{} | {}, 어서오세요!".format(message.author, message.author.mention))

#     # DM 보내기 시도
#     # await message.author.send("{} | {} 유저님, 환영합니다.".format(message.author, message.author.mention))
#     try:
#         await message.author.send(f"{message.author} | {message.author.mention} 유저님, 환영합니다.")
#     except discord.Forbidden:
#         print(f"[DM 불가] {message.author} 님이 DM을 받을 수 없습니다.")
#     except discord.HTTPException as e:
#         print(f"[DM 에러] {message.author} : {e}")

# 아래 코드들은 client.event의 on_message를 주석 처리하고 실행

@client.command(aliases=['hi'])
async def hello(ctx):
    await ctx.send("안녕하세요!")

@client.command(aliases=['로그인', '접속하기'])
async def login(ctx):
    await ctx.channel.send("{} | {}님, 어서오세요!".format(ctx.author, ctx.author.mention))




client.run(Token)

"""
실행 시 오류:
discord.errors.PrivilegedIntentsRequired: ... requesting privileged intents that have not been explicitly enabled ...
------> 권한이 필요한 intent(privileged intents)를 요청했는데,
Discord 개발자 페이지에서 해당 intent를 활성화하지 않을 것!

해결법: Discord 개발자 포털로 이동 -> 사용하는 봇 이름 클릭
-> "Bot"탭 클릭 -> “Privileged Gateway Intents”에서 필요한 것 켜기
- PRESENCE INTENT (상태 정보 감지)
- SERVER MEMBERS INTENT (서버 멤버 감지)
- MESSAGE CONTENT INTENT (메시지 내용 읽기 — 명령어 인식용)
"""


"""
실행 시 오류:
discord.errors.HTTPException: 400 Bad Request (error code: 50007): Cannot send messages to this user
------> 해당 유저가 DM을 비활성화했거나, 봇이 유저와 친구가 아니거나, DM을 차단한 유저에게 메세지를 보낼 때

해결법: try/except 넣기
    try:
        await message.author.send(f"{message.author} | {message.author.mention} 유저님, 환영합니다.")
    except discord.Forbidden:
        print(f"[DM 불가] {message.author} 님이 DM을 받을 수 없습니다.")
    except discord.HTTPException as e:
        print(f"[DM 에러] {message.author} : {e}")
"""

"""
실행 시 오류: 400 Bad Request (error code: 50007): Cannot send messages to this user
------> 위에 경우가 아닌데도 뜸 -> 봇이 보내는 메세지도 함수에 들어가는 게 문제!(봇이 자기자신에게 메세지 보내는 건 좀 이상하지)

해결법: 봇 자신의 메세지만 무시처리
    if message.author == client.user:
        return  # 봇 자신의 메시지 무시
"""