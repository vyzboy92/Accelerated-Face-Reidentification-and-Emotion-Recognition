#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import face_recognition
import cv2
import numpy as np
import pymongo
import sys
import time
import logging as log
from imutils.video import WebcamVideoStream
from openvino.inference_engine import IENetwork, IEPlugin
from sklearn.metrics.pairwise import cosine_similarity
import requests
import subprocess
import datetime as dt
from multiprocessing import Process, Queue
from imutils import face_utils
import dlib
import json
import socket
import base64
import pickle
import jsonpickle
import zmq
# from websocket_server import WebsocketServer
from websocket import create_connection

em_client = pymongo.MongoClient("mongodb://localhost:27017/")
dblist = em_client.list_database_names()
if "Main_DB" in dblist:
    print("========================")
    print("Main_db found in Mongo")
    print("========================")
em_db = em_client["Main_DB"]
em_col = em_db["face_info"]
prev_face_col = em_db["face_logs"]

# run docker and connect to api
subprocess.call("./docker_run.sh")
addr = 'http://localhost:5000'
test_url = addr + '/api/test'

# Web-socket connection
ws = create_connection("ws://localhost:8888/ws")

q = Queue(maxsize=100)


# emotion task scheduler
def process_task():
    global test_url, em_col

    while True:
        item = q.get()
        if item['eof'] is True:
            print("break")
            break
        elif item is None:
            continue
        name = item['face_id']
        faces = item['frame']
        filename = '/home/vysakh/Accubits/INTEL/Accelerated-Face-Reidentification-and-Emotion-Recognition' \
                   '/docker_fetch/image' + str(name) + '.jpg'
        cv2.imwrite(filename, faces)
        # get response
        params = {'arg1': name}
        response = requests.post(test_url, data=params)
        print(response.text)

        del response, item

    q.put({'eof': True})


# container function to initialise OpenVINO models
def init_model(xml, bins):
    model_xml = xml
    model_bin = bins
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension(
        '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_nets = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_nets, n, c, w, h, input_blob, out_blob, plugin

def send_message(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")

def calculate_cosine(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    sims = cosine_similarity(face_encodings, face_to_compare.reshape(1, -1))
    return sims


def main():
    emotion_list = ['neutral', 'happy', 'sad', 'surprise', 'anger']  # for parsing emotion detection result
    gender_list = ['female', 'male']  # parse gender model result

    # paths to models
    face_xml = "utils/face-detection-adas-0001.xml"
    face_bin = "utils/face-detection-adas-0001.bin"
    emotion_xml = "utils/emotions-recognition-retail-0003.xml"
    emotion_bin = "utils/emotions-recognition-retail-0003.bin"
    age_gender_xml = "utils/age-gender-recognition-retail-0013.xml"
    age_gender_bin = "utils/age-gender-recognition-retail-0013.bin"
    landmark_model = "utils/shape_predictor_68_face_landmarks.dat"

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    exec_net, n, c, w, h, input_blob, out_blob, plugin = init_model(emotion_xml, emotion_bin)
    age_gender_net, n_a, c_a, w_a, h_a, input_blob_a, out_blob_a, plugin_a = init_model(age_gender_xml, age_gender_bin)
    face_detection_net, n_f, c_f, w_f, h_f, input_blob_f, out_blob_f, plugin_f = init_model(face_xml, face_bin)
    landmark_predictor = dlib.shape_predictor(landmark_model)

    # load known faces from DB
    faces = list(em_col.find({}))
    # Get a reference to webcam #0 (the default one)
    # fvs = WebcamVideoStream(src='rtsp://admin:AccubitsEmotix@192.168.0.10:554/Streaming/channels/1/').start()
    fvs = WebcamVideoStream(src=0).start()
    time.sleep(0.5)
    known_face_encodings = []
    known_face_names = []

    # Create arrays of known face encodings and their names
    for face in faces:
        for face_encods in face['encoding']:
            known_face_encodings.append(np.asarray(face_encods))
            known_face_names.append(face['name'])

    # Initialize some variables
    frame_count = 0
    cur_request_id = 0
    next_request_id = 1
    cur_request_id_a = 0
    next_request_id_a = 1
    cur_request_id_f = 0
    next_request_id_f = 1
    emotion = None
    initial_frame = fvs.read()
    initial_h, initial_w = initial_frame.shape[:2]
    while True:
        # Grab a single frame of video
        frame = fvs.read()
        frame_copy = fvs.read()
        if frame is None:
            break

        # Find all the faces and face encodings in the current frame of video

        face_locations = []
        face_locations_keypoint = []
        in_frame = cv2.resize(frame, (w_f, h_f))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n_f, c_f, h_f, w_f))
        face_detection_net.start_async(request_id=cur_request_id_f, inputs={input_blob: in_frame})
        if face_detection_net.requests[cur_request_id_f].wait(-1) == 0:
            face_detection_res = face_detection_net.requests[cur_request_id_f].outputs[out_blob_f]
            for face_loc in face_detection_res[0][0]:
                if face_loc[2] > 0.5:
                    xmin = abs(int(face_loc[3] * initial_w))
                    ymin = abs(int(face_loc[4] * initial_h))
                    xmax = abs(int(face_loc[5] * initial_w))
                    ymax = abs(int(face_loc[6] * initial_h))
                    face_locations.append((xmin, ymin, xmax, ymax))
                    face_locations_keypoint.append(dlib.rectangle(xmin, ymin, xmax, ymax))

        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)
        emotion_payload = []
        # Display the results
        for (left, top, right, bottom), name in zip(face_locations, face_names):

            face = frame[top:bottom, left:right]  # extract face
            # run the emotion inference on extracted face
            in_frame = cv2.resize(face, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                emotion_dict = dict()
                emotion_dict['neutral'] = str(res[0][0][0][0])
                emotion_dict['happy'] = str(res[0][1][0][0])
                emotion_dict['sad'] = str(res[0][2][0][0])
                emotion_dict['surprise'] = str(res[0][3][0][0])
                emotion_dict['angry'] = str(res[0][4][0][0])
                emotion_payload.append(emotion_dict)
                emo_pred = np.argmax(res)
                emotion = emotion_list[emo_pred]
            # run age and gender inference
            age_frame = cv2.resize(face, (w_a, h_a))
            age_frame = age_frame.transpose((2, 0, 1))
            age_frame = age_frame.reshape((n_a, c_a, h_a, w_a))
            age_gender_net.start_async(request_id=cur_request_id_a, inputs={input_blob_a: age_frame})
            if age_gender_net.requests[cur_request_id_a].wait(-1) == 0:
                dec = age_gender_net.requests[cur_request_id_a].outputs
                gender = dec['prob']
                age = dec['age_conv3']
                age = int(age[0][0][0][0] * 100)
                gender = gender_list[np.argmax(gender)]
            # add face to queue for emotyx module
            if frame_count % 100 == 0:
                _, face_id = str(dt.datetime.now()).split('.')
                face_pic = frame_copy[top - 100:bottom + 100, left - 100:right + 100]
                item = dict()
                item['frame'] = face_pic
                item['face_id'] = face_id
                item['eof'] = False
                q.put(item)
            if name is not "Unknown":
                if not list(prev_face_col.find({"name": name})):
                    prev_face_col.insert(
                        {'name': name, 'last_seen': dt.datetime.now(), 'image': 'face_logs/' + str(name) + '.jpg'})
                else:
                    prev_face_col.update({'name': name}, {'$set': {'last_seen': dt.datetime.now()}})
                cv2.imwrite('face_logs/' + str(name) + '.jpg', face)
            overlay = frame.copy()
            alpha = 0.6
            cv2.rectangle(overlay, (left, top), (right, bottom), (65, 65, 65), 2)
            cv2.rectangle(overlay, (right, top), (right + 150, top + 100), (65, 65, 65), cv2.FILLED)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha,
                            0, frame)
            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # cv2.rectangle(frame, (right, top), (right + 150, top + 100), (0, 125, 125), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Name: ' + name, (right + 5, top + 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'Emotion: ' + emotion, (right + 5, top + 40), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'Gender: ' + gender, (right + 5, top + 60), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'Age: ' + str(age), (right + 5, top + 80), font, 0.5, (255, 255, 255), 1)
        for loc in face_locations_keypoint:
            shape = landmark_predictor(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY), loc)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 125, 125), -1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        frame_count += 1
        cur_request_id, next_request_id = next_request_id, cur_request_id
        cur_request_id_a, next_request_id_a = next_request_id_a, cur_request_id_a
        cur_request_id_f, next_request_id_f = next_request_id_f, cur_request_id_f

        # send image payload to socket
        _, frame_to_send = cv2.imencode('.jpg', frame, encode_param)
        jpg_as_text = base64.b64encode(frame_to_send)
        emotion_content = json.dumps({'status': 'success', 'data': emotion_payload, 'type': 'dict'})
        image_content = json.dumps({'status': 'success', 'data': str(jpg_as_text), 'type': 'image'})
        ws.send(emotion_content)
        ws.send(image_content)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    q.put({'eof': True})
    # Release handle to the webcam
    fvs.stop()
    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    jobs = []
    buff_dict = dict()
    number_of_Threads = 2
    for i in range(number_of_Threads):
        t = Process(target=process_task)
    jobs.append(t)
    t.start()
    main()



