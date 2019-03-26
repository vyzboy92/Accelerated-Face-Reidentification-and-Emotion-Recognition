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

em_client = pymongo.MongoClient("mongodb://localhost:27017/")
dblist = em_client.list_database_names()
if "Main_DB" in dblist:
    print("========================")
    print("Main_db found in Mongo")
    print("========================")
em_db = em_client["Main_DB"]
em_col = em_db["face_info"]


def init_emotion():
    model_xml = "/home/iti/Accubits/Intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
    model_bin = "/home/iti/Accubits/Intel/computer_vision_sdk_2018.5.455/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin"
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension('/opt/intel/computer_vision_sdk/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so')
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
    exec_net = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_net, n, c, w, h, input_blob, out_blob, plugin


def main():
    emotion_list = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    exec_net, n, c, w, h, input_blob, out_blob, plugin = init_emotion()
    faces = list(em_col.find({}))
    # Get a reference to webcam #0 (the default one)
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
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cur_request_id = 0
    next_request_id = 1
    emotion = None
    while True:
        # Grab a single frame of video
        frame = fvs.read()
        if frame is None:
            break
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

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

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            face = frame[top:bottom, left:right]
            cv2.imshow('face', face)
            in_frame = cv2.resize(face, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                emo_pred = np.argmax(res)
                emotion = emotion_list[emo_pred]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, emotion, (left + 6, bottom + 12), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', cv2.resize(frame, (1280, 720)))
        cur_request_id, next_request_id = next_request_id, cur_request_id
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    # video_capture.release()
    fvs.stop()
    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    main()
