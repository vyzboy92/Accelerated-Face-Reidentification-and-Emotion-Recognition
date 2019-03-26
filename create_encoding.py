import face_recognition
import cv2
import argparse
import pymongo

em_client = pymongo.MongoClient("mongodb://localhost:27017/")
dblist = em_client.list_database_names()
if "Main_DB" in dblist:
    print("========================")
    print("Main_db found in Mongo")
    print("========================")
em_db = em_client["Main_DB"]
em_col = em_db["face_info"]  # logs face encodings as arrays with corresponding name

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True,
                help="path to input image")
ap.add_argument("-n", "--name", required=True,
                help="name of user", type=str)
args = vars(ap.parse_args())
image = face_recognition.load_image_file(args["input_image"])
face_encoding = face_recognition.face_encodings(image)[0]
if em_col.find_one({"name": args["name"]}) is not None:
    em_col.update({'name': args["name"]}, {'$push': {'encoding': list(face_encoding)}})
else:
    data = {'name': args["name"], 'encoding': [list(face_encoding)]}
    em_col.insert(data)

