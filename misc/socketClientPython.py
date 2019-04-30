from websocket import create_connection

ws = create_connection("ws://localhost:8888/ws")
ws.send("Python , Vysah")

# while True:
#     print ws.recv()


