import socket
import sys
import cv2
import pickle
import numpy as np
import struct  ## new
import zlib
import json

HOST = ''
PORT = 8486

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(1)
print('Socket now listening')
conn, addr = s.accept()
while True:
    dats = conn.recv(4096)
    # frame = pickle.loads(dats)
    # cv2.imshow('fr', frame)
    print(dats)
    print('-----')
