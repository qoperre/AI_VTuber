#  get_vts_token.py





# =========================================
#             Made by GPT
# =========================================

import asyncio
import websockets
import json

VTS_URL = "ws://localhost:8001"
PLUGIN_NAME = "Discord AI VTuber"
PLUGIN_AUTHOR = "Qoperre"
PLUGIN_VERSION = "1.0"

async def main():
    async with websockets.connect(VTS_URL) as ws:
        # 인증 요청 보내기
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth_request",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_AUTHOR,
                "pluginVersion": PLUGIN_VERSION
            }
        }

        await ws.send(json.dumps(req))
        print("🔗 인증 요청 전송. VTube Studio에서 승인 창이 뜰 거야!")

        # 응답 수신
        resp = await ws.recv()
        data = json.loads(resp)
        print("\n📨 응답:", json.dumps(data, indent=2, ensure_ascii=False))

        token = data.get("data", {}).get("authenticationToken")
        if token:
            with open("vts_token.json", "w", encoding="utf-8") as f:
                json.dump({"token": token}, f)
            print(f"\n✅ 토큰 발급 완료! 저장됨: {token}")
        else:
            print("❌ 토큰 발급 실패! 승인 안 했을 가능성이 있어.")

asyncio.run(main())
