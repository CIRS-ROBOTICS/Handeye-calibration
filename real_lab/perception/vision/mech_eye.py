import numpy as np
from scipy.spatial.transform import Rotation as R

from .camera import Camera
from robot_tool.perception.vision.mech_eye_api import CameraClient


DEFAULT_MECHEYE_CAMERA_ROTATION = np.array([-0.998139915692, -0.0137152080259, 0.058834506008, 0.00819162216657])
DEFAULT_MECHEYE_CAMERA_TRANSLATION = np.array([[0.266036682754], [-0.563285940841], [1.14197299431]])


class MechEye(Camera):
    def __init__(self, ip, cam_rotation=None, cam_translation=None):
        if cam_rotation is None:
            cam_rotation = DEFAULT_MECHEYE_CAMERA_ROTATION
        if cam_translation is None:
            cam_translation = DEFAULT_MECHEYE_CAMERA_TRANSLATION

        r = R.from_quat(cam_rotation)
        cam_rotation = r.as_matrix()
        self.cam_pose = np.zeros((4, 4), dtype=np.float32)
        self.cam_pose[:3, :3] = cam_rotation
        self.cam_pose[:3, 3:] = cam_translation
        self.cam_pose[-1, -1] = 1

        self.ip = ip
        self.cam = None
        self.cam_intri = np.zeros([3, 3])

        self.cam = CameraClient.CameraClient()
        self.cam.connect(self.ip)
        cam_intri = self.cam.getCameraIntri()
        self.cam_intri[0, 0] = cam_intri[0]
        self.cam_intri[1, 1] = cam_intri[1]
        self.cam_intri[0, 2] = cam_intri[2]
        self.cam_intri[1, 2] = cam_intri[3]
        self.cam_intri[2, 2] = 1

        self._cam_name = "Mech-Eye"

    def get_data(self):
        return self.cam.captureColorImg(), self.cam.captureDepthImg()

    def close(self):
        pass

    def get_camera_params(self):
        return {"image_type": "bgr",
                "cam_intri": self.cam_intri.copy(),
                "cam_pose": self.cam_pose.copy(),
                "cam_name": self._cam_name
                }
