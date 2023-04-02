import pyzed.sl as sl
import numpy as np
from scipy.spatial.transform import Rotation as R

from .camera import Camera

DEFAULT_ZED_CAMERA_ROTATION = np.array([-0.874125361082, -0.0270816605365, -0.00440328852424, 0.484924785742])
DEFAULT_ZED_CAMERA_TRANSLATION = np.array([[-0.0130321939147], [-1.22888842636], [0.479256365879]])


class ZED(Camera):
    def __init__(self, cam_rotation=None, cam_translation=None, init_params=None):
        if cam_rotation is None:
            cam_rotation = DEFAULT_ZED_CAMERA_ROTATION
        if cam_translation is None:
            cam_translation = DEFAULT_ZED_CAMERA_TRANSLATION

        r = R.from_quat(cam_rotation)
        cam_rotation = r.as_matrix()
        self.cam_pose = np.zeros((4, 4), dtype=np.float32)
        self.cam_pose[:3, :3] = cam_rotation
        self.cam_pose[:3, 3:] = cam_translation
        self.cam_pose[-1, -1] = 1

        self.cam_intri = np.zeros([3, 3])

        self.init_params = init_params
        self.runtime_params = None
        self.cam = sl.Camera()
        if self.init_params is None:
            init_params = sl.InitParameters()

            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
            init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
            init_params.camera_resolution = sl.RESOLUTION.HD720
            self.init_params = init_params

        err = self.cam.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        runtime_parameters.confidence_threshold = 100
        runtime_parameters.textureness_confidence_threshold = 100

        self.runtime_params = runtime_parameters

        cam_intri = self.cam.get_camera_information().calibration_parameters.left_cam
        self.cam_intri[0, 0] = cam_intri.fx
        self.cam_intri[1, 1] = cam_intri.fy
        self.cam_intri[0, 2] = cam_intri.cx
        self.cam_intri[1, 2] = cam_intri.cy
        self.cam_intri[2, 2] = 1

        self._cam_name = "ZED"

    def get_data(self):
        image = sl.Mat()
        depth = sl.Mat()

        if self.cam.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.cam.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            self.cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

        return image.get_data(), depth.get_data()

    def close(self):
        self.cam.close()

    def get_camera_params(self):
        return {"image_type": "bgr",
                "cam_intri": self.cam_intri.copy(),
                "cam_pose": self.cam_pose.copy(),
                "cam_name": self._cam_name
                }