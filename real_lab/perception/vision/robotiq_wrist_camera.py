import numpy as np
import requests

from .camera import Camera


class RobotiqWristCamera(Camera):
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip

        self._cam_name = "Robotiq Wrist Camera"
        self.cam_intri = None
        self.cam_pose = None

    def get_data(self):
        """

        :return: Color Image with BGR style, Depth Image with Meters
        """
        try:
            resp = requests.get("http://" + self.robot_ip + ":4242/current.jpg?type=color").content
        except:
            return None

        return np.asarray(bytearray(resp), dtype="uint8")

    def close(self):
        """

        :return:
        """
        pass

    def get_camera_params(self):
        """

        :return:
        """
        return {"image_type": "bgr",
                "cam_intri": self.cam_intri.copy(),
                "cam_pose": self.cam_pose.copy(),
                "cam_name": self._cam_name
                }
