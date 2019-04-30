from flask import Flask, request, Response
import subprocess
import csv
import socket
import json

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('172.17.0.1', 8485))
    arg1 = request.form['arg1']
    # call affectiva
    subprocess.call(['bash', 'detect.sh', str(arg1)])

    # read the info csv
    csvfile = open('/tmp/image/image' + str(arg1) + '.csv', 'r')
    reader = csv.DictReader(csvfile)
    for row in reader:
        aff_dict = row

    # build a response dict to send back to client & encode response using jsonpickle
    response_pickled = json.dumps(aff_dict).encode('utf-8')
    client_socket.sendall(response_pickled)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
