#!/usr/bin/env python
import zmq
from . import image_pb2
from . import cameraStatus_pb2
import sys

def createAddr(ip, port):
    return "tcp://" + ip + ":" + str(port)


class ZmqClient:
    def __init__(self):
        self.__addr = ""
        self.__reqBuf = ""
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REQ)

    def setAddr(self, ip, port, rcvTimeout):
        if len(ip) == 0:
            return False
        if len(self.__addr) != 0:
            self.__socket.disconnect()
        self.__socket.setsockopt(zmq.RCVTIMEO, rcvTimeout)
        self.__socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__addr = createAddr(ip, port)
        print("connect to %s" % (self.__addr))
        self.__socket.connect(self.__addr)
        return True

    def disconnect(self):
        self.__socket.disconnect(self.__addr)
        self.__addr = ""

    def empty(self):
        return len(self.__addr) == 0

    def sendReq(self, request):
        try:
            self.__reqBuf = request.SerializeToString()
            if len(self.__addr) == 0:
                return {}
            received = self.__sendMsg()
            if len(received) == 0:
                return {}
            reply = image_pb2.Response()

            reply.ParseFromString(received)
            return reply
        except Exception as e:
            print("Network Error! Please check your ip address and connection!")
            sys.exit(0)

    def __sendMsg(self):
        try:
            message = self.__reqBuf
            self.__socket.send(message)
            return self.__socket.recv()
        except Exception as e:
            print("Network Error! Please check your ip address and connection!")
            sys.exit(0)
