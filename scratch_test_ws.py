import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket.")
            for i in range(3):
                msg = await ws.recv()
                data = json.loads(msg)
                print(f"Received tick: {data.get('tick')}")
    except Exception as e:
        print(f"WebSocket Error: {e}")

asyncio.run(test())
