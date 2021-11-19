import asyncio
import websockets

async def main():
    uri = "ws://67.20.204.23:8765"
    async with websockets.connect(uri) as websocket:
        path = input('File Path: ')
                
        with open(path) as file:
            data = file.read()
            await websocket.send(data)

        message = await websocket.recv()
        print(f"<<< {message}")
        
        print('Waiting for server to send file')
        
        while True:
            file = await websocket.recv()
            print(len(file))
            
        message = await websocket.send('Received')
        print(f'<<< File of type {type(data)} received from server')

asyncio.run(main())