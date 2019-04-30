import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web

'''
This is a simple Websocket Echo server that uses the Tornado websocket handler.
Please run `pip install tornado` with python of version 2.7.9 or greater to install tornado.
This program will echo back the reverse of whatever it recieves.
Messages are output to the terminal for debuggin purposes. 
'''


class WSHandler(tornado.websocket.WebSocketHandler):
    connections = set()

    def open(self):
        self.connections.add(self)
        print ('new connection')

    def on_message(self, message):
        print ('message received:  %s' % message)
        # self.write_message(message[::-1])
        [con.write_message(message) for con in self.connections]

    def on_close(self):
        self.connections.remove(self)
        print ('connection closed')

    def check_origin(self, origin):
        return True


application = tornado.web.Application([
    (r'/ws', WSHandler),
])

if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    # myIP = socket.gethostbyname(socket.gethostname())
    myIP = 'localhost'
    print ('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()