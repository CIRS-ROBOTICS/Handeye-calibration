from openni import openni2
from openni import _openni2 as _c_api

import numpy as np
from collections import Iterable
from scipy.spatial.transform import Rotation as R
import cv2
import time

from .camera import Camera
from robot_tool.utils.logger import warn, error

DEFAULT_XTION_CAMERA_INTRINSICS = np.array([[570.3422241210938, 0., 319.5], [0., 570.3422241210938, 239.5], [0., 0., 1.]])

DEFAULT_XTION_CAMERA_ROTATION = np.array([-0.998764697267, -0.0287046450375, 0.-0.0150774172793, 0.0376536098512])
DEFAULT_XTION_CAMERA_TRANSLATION = np.array([[0.0509080769608], [-0.881967529616], [1.08975274721]])


class XtionPro(Camera):
    def __init__(self, cam_rotation=None, cam_translation=None):
        super(XtionPro, self).__init__()
        self.height, self.width = 480, 640
        self.fps = 30
        self.rgb_stream, self.depth_stream = self._init_cam()

        r = R.from_quat(DEFAULT_XTION_CAMERA_ROTATION)
        cam_rotation = r.as_matrix()
        self.cam_pose = np.zeros((4, 4), dtype=np.float32)
        self.cam_pose[:3, :3] = cam_rotation
        self.cam_pose[:3, 3:] = DEFAULT_XTION_CAMERA_TRANSLATION
        self.cam_pose[-1, -1] = 1
        self.cam_intri = DEFAULT_XTION_CAMERA_INTRINSICS

    def get_camera_params(self):
        pass

    def get_data(self):
        return self._get_rgb_data(bgr=False), self._get_depth_data()

    def close(self):
        self.rgb_stream.stop()
        self.depth_stream.stop()
        openni2.unload()

    def _get_rgb_data(self, bgr, flip=1):
        frame = self.rgb_stream.read_frame()
        img_buffer = frame.get_buffer_as_triplet()
        img = np.array(img_buffer).reshape([self.height, self.width, 3])

        if bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.flip(img, flip)

        return img

    def _get_depth_data(self, flip=1):
        frame = self.depth_stream.read_frame()
        img_buffer = frame.get_buffer_as_uint16()
        img = np.ndarray((self.height, self.width), dtype=np.uint16, buffer=img_buffer)

        img = cv2.flip(img, flip)
        return img

    def _init_cam(self):
        openni2.initialize()

        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        rgb_stream = dev.create_color_stream()

        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        rgb_stream.set_video_mode(_c_api.OniVideoMode(pixelFormat=_c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                      resolutionX=self.width,
                                                      resolutionY=self.height,
                                                      fps=self.fps))
        depth_stream.set_video_mode(_c_api.OniVideoMode(pixelFormat=_c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                        resolutionX=self.width,
                                                        resolutionY=self.height,
                                                        fps=self.fps))

        rgb_stream.start()
        depth_stream.start()
        time.sleep(1)

        return rgb_stream, depth_stream
