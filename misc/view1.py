import cv2
import zmq
import base64
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5556')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        frame = footage_socket.recv_string()
        print(frame)

    except KeyboardInterrupt:
        break