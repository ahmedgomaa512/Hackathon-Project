import cv2
import socket
import pickle
import struct

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))
server.listen(1)

print("Waiting for client...")
conn, addr = server.accept()
print("Connected:", addr)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    conn.sendall(message)