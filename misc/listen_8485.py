import socket
import sys
import cv2
import pickle
import numpy as np
import struct  ## new
import zlib
import json

HOST = ''
PORT = 8485

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(1)
print('Socket now listening')

while True:
    conn, addr = s.accept()
    dats = conn.recv(4096)
    frame = json.loads(dats.decode('utf-8'))
    print(frame)
    print('-----')
