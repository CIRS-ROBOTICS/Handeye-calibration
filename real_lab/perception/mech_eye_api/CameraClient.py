#!/usr/bin/env python
from .ZmqClient import ZmqClient
from . import image_pb2
from . import cameraStatus_pb2
import cv2
import numpy as np
import open3d
import ctypes
from struct import unpack

Encode32FBias = 32768
SIZE_OF_DOUBLE = ctypes.sizeof(ctypes.c_double)


def find_last_of(string, c):
    last_position = -1
    while True:
        position = string.find(c, last_position + 1)
        if position == -1:
            return last_position
        last_position = position


class ImageType():
    DEPTH = 1
    COLOR = 2
    COLOR_DEPTH = COLOR | DEPTH


class NetCamCmd():
    CaptureImage = 0
    GetCameraIntri = 11
    GetCameraStatus = 19
    SetCameraParameter = 25
    GetCameraParameter = 26
    CaptureGratingImage = 8


class CameraIntri:
    def __init__(self):
        self.__fx = 0.0
        self.__fy = 0.0
        self.__u = 0.0
        self.__v = 0.0

    def isZero(self):
        return (self.__fx == 0.0) and (self.__fy == 0.0) and (self.__u == 0.0) and (self.__v == 0.0)

    def setValue(self, fx, fy, u, v):
        self.__fx = fx
        self.__fy = fy
        self.__u = u
        self.__v = v

    def getValue(self):
        return self.__fx, self.__fy, self.__u, self.__v


def readDouble(data, pos):
    if pos + SIZE_OF_DOUBLE > len(data):
        return 0
    strFromQDataStream = data[pos: pos + SIZE_OF_DOUBLE]
    v = unpack(">d", strFromQDataStream)
    return v


def asMat(data, offset=0):
    mat = []
    for i in range(offset, len(data)):
        mat.append(data[i]) # in Python 2.x you may need to unpack data: unpack(">B",data[i])
    mat = np.asarray(mat, dtype="uint8")
    return mat

def matC1ToC3(mat):
    if len(mat) == 0:
        return []
    if len(np.shape(mat)) != 2 or np.shape(mat)[0]%3 != 0:
        return []
    rows, cols = np.shape(mat)
    rel = []
    rel.append(mat[0:int(rows/3),:])
    rel.append(mat[int(rows/3):int(2*rows/3),:])
    rel.append(mat[int(2*rows/3):rows,:])
    rel = cv2.merge(rel)
    return rel


def read32FC3Mat(data):
    if len(data) == 0:
        return []
    scale = readDouble(data,0)[0]
    matC1 = cv2.imdecode(asMat(data,SIZE_OF_DOUBLE),cv2.IMREAD_ANYDEPTH)
    bias16UC3 = matC1ToC3(matC1)
    t = np.float32(bias16UC3)
    mat32F = (t - Encode32FBias)/scale
    return mat32F

def read32FC1Mat(data, offset=0):
    if len(data) == 0:
        return []
    scale = readDouble(data, offset)
    bias16U = cv2.imdecode(asMat(data, SIZE_OF_DOUBLE + offset), cv2.IMREAD_ANYDEPTH)
    bias32F = bias16U.astype(np.float32)
    mat32F = bias32F
    mat32F[:, :] -= Encode32FBias
    if scale == 0:
        return cv2.Mat()
    else:
        return mat32F / scale


class CameraClient(ZmqClient):
    __kImagePort = 5577

    def __init__(self):
        ZmqClient.__init__(self)

    def connect(self, ip):
        return ZmqClient.setAddr(self, ip, self.__kImagePort, 60000)

    def captureDepthImg(self):
        response = self.__sendRequest(NetCamCmd.CaptureImage, ImageType.DEPTH)
        if len(response.imageDepth) == 0:
            print("Client depth image is empty!")
            return {}
        # print("Depth image captured!")
        return read32FC1Mat(response.imageDepth, 2)

    def captureColorImg(self):
        response = self.__sendRequest(NetCamCmd.CaptureImage, ImageType.COLOR)
        if len(response.imageRGB) == 0:
            print("Client depth image is empty!")
            return {}
        # print("Color image captured!")
        return cv2.imdecode(asMat(response.imageRGB), cv2.IMREAD_COLOR)

    def getCameraIntri(self):
        response = self.__sendRequest(NetCamCmd.GetCameraIntri, 0)
        print("Camera intrinsics: ")
        print(response.camIntri)

        start = find_last_of(response.camIntri, "[")
        end = find_last_of(response.camIntri, "]")
        if start == -1 or end == -1 or end < start:
            print("Wrong camera intrinsics.")
            return {}

        intriStr = response.camIntri[start + 1: end]
        intriValue = intriStr.split(",")
        if len(intriValue) != 4:
            print("Wrong intrinscis value")
            return {}

        intri = CameraIntri()
        intri.setValue(
            float(
                intriValue[0]), float(
                intriValue[1]), float(
                intriValue[2]), float(
                    intriValue[3]))

        intriValueDouble = intri.getValue()
        # print("fx = %.13f" % (intriValueDouble[0]))
        # print("fy = %.13f" % (intriValueDouble[1]))
        # print("u = %.13f" % (intriValueDouble[2]))
        # print("v = %.13f" % (intriValueDouble[3]))
        return intriValueDouble

    def getCameraId(self):
        return self.__getCameraStatus().eyeId

    def getCameraIp(self):
        return self.__getCameraStatus().ip

    def getCameraVersion(self):
        return self.__getCameraStatus().version

    def getParameter(self, paraName):
        request = image_pb2.Request()
        request.command = NetCamCmd.GetCameraParameter
        request.valueString = paraName
        reply = ZmqClient.sendReq(self, request)
        return float(reply.parameterValue)

    def setParameter(self, paraName, value):
        request = image_pb2.Request()
        request.command = NetCamCmd.SetCameraParameter
        request.valueString = paraName
        request.valueDouble = value
        reply = ZmqClient.sendReq(self, request)
        return reply.error

    def __sendRequest(self, commandx, value):
        request = image_pb2.Request()
        request.command = commandx
        request.valueDouble = value
        reply = ZmqClient.sendReq(self, request)
        return reply

    def __getCameraStatus(self):
        reply = self.__sendRequest(NetCamCmd.GetCameraStatus, 0)
        StatusUnicode = reply.cameraStatus
        StatusBytes = StatusUnicode.encode("utf-8")
        cameraStatus = cameraStatus_pb2.CameraStatus()
        cameraStatus.ParseFromString(StatusBytes)
        return cameraStatus

    def captureCloud(self):
        reply = self.__sendRequest(NetCamCmd.CaptureGratingImage, 4)
        depthC3 = read32FC3Mat(reply.imageGrating)
        color = self.captureColorImg()
        return self.getRGBCloud(color, depthC3)
    
    def getRGBCloud(self, color, depth):
        test_pcd = open3d.geometry.PointCloud()  # 定义点云
        color.flatten()
        color = color / 256
        color.resize(int(np.size(color)/3),3)
        depth.flatten()
        depth = depth * 0.001
        depth.resize(int(np.size(depth)/3),3)
        test_pcd.points = open3d.utility.Vector3dVector(depth)  # 定义点云坐标位置
        test_pcd.colors = open3d.utility.Vector3dVector(color)  # 定义点云的颜色
        return test_pcd
