import os
import time

import asyncio
import websockets

async def hello(websocket, path):
    data = await websocket.recv()
    print(f'<<< File of type {type(data)} received from client')
    
    await websocket.send('File received at server')
    
    path = input('File path: ')
    file_size = os.path.getsize(path)
    
    with open(path, encoding = "ISO-8859-1") as file:
        c = 0
        start_time = time.time()

        while c <= file_size:
            data = file.read(1024)
            if not (data):
                break
            await websocket.send(data)
            c += len(data)
    
    
#     with open(path, encoding = "ISO-8859-1") as file:
#         data = file.read()
#         print('File Read, seding to client')
#         await websocket.send(data)
#         print('File sent')
        
    print('File sent')
    message = await websocket.recv()
    print(f'<<< {message}')

async def main():
    async with websockets.serve(hello, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())