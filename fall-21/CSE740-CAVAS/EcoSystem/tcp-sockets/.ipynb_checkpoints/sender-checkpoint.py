import os
import socket
import time

# Create a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('0.0.0.0', 22222))
sock.listen(5)
print('Host IP: ', sock.getsockname())

# Accept Connection
client, addr = sock.accept()

# Read file metadata
file_name = input('File Path:')
file_size = os.path.getsize(file_name)

# Send file metadata
client.send(file_name.encode())
client.send(str(file_size).encode())

# Read file and send data
with open(file_name, 'rb') as file:
    c = 0
    start_time = time.time()

    while c <= file_size:
        data = file.read(1024)
        if not (data):
            break
        client.sendall(data)
        c += len(data)

    end_time = time.time()

print('Time taken: ', end_time - start_time)
