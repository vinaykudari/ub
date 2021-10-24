import os
import socket
import time

host = input('Host Node IP: ')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to socket
try:
    sock.connect((host, 22222))
    print('Connected')
except:
    print('Host Not Found')
    exit(0)

# Prase file metadata
file_name = sock.recv(100).decode()
file_size = sock.recv(100).decode()

print(file_name)
print(file_size)

# Write to a local file
with open('received_file', 'wb') as file:
    c = 0
    # Starting the time capture.
    start_time = time.time()

    while c <= int(file_size):
        data = sock.recv(1024)
        if not (data):
            break
        file.write(data)
        c += len(data)

    end_time = time.time()

print('File Received')
print('Time taken: ', end_time - start_time)
