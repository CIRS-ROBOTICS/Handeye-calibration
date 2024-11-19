import pyrealsense2 as rs

import numpy as np
from collections import Iterable
from scipy.spatial.transform import Rotation as R

from .camera import Camera
# from robot_tool.utils.logger import warn, error

DEFAULT_REALSENSE_CAMERA_ROTATION = np.array([-0.874125361082, -0.0270816605365, -0.00440328852424, 0.484924785742])
DEFAULT_REALSENSE_CAMERA_TRANSLATION = np.array([[-0.0130321939147], [-1.22888842636], [0.479256365879]])


class RealSense(Camera):
    def __init__(self, cam_rotation=None, cam_translation=None, init_params=None, post_process=None, model=None):
        if cam_rotation is None:
            cam_rotation = DEFAULT_REALSENSE_CAMERA_ROTATION
        if cam_translation is None:
            cam_translation = DEFAULT_REALSENSE_CAMERA_TRANSLATION

        self.pipeline = rs.pipeline()  # 声明通道，用于封装设备传感器

        if init_params is None:
            init_params = {}
            config = rs.config()
            if model == "SR300":  # 根据设备选型
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            else:
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            init_params['config'] = config

            # align depth to color
            init_params['align'] = rs.stream.color

        self.init_params = init_params
        self.aligned_frames = rs.align(init_params['align'])
        config_params = self.init_params['config']
        profile = self.pipeline.start(config_params) # 创建用户配置文件

        # color_sensor = profile.get_device().first_color_sensor() # 获得RGB传感器
        depth_sensor = profile.get_device().first_depth_sensor() # 获得深度传感器
        # For different situation, set different depth_units,
        # since different depth_units lead to different valid depth range.
        # like, for depth_units is 1e-6, range is 0.5 ~ 0.62 meters.
        if model == "SR300":
            # depth_sensor.set_option(rs.option.depth_units, 0.000125)
            pass
        else:
            depth_sensor.set_option(rs.option.depth_units, 0.0001) # Changing the depth units to 100μm (0.0001m) allows for the max  measured range ~6.4m

        self.depth_scale = depth_sensor.get_depth_scale()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.aligned_frames.process(frames) # 获得对齐帧
        depth_frame = aligned_frames.get_depth_frame() # 深度对齐帧
        color_frame = aligned_frames.get_color_frame() # RGB对齐帧
        # 让 RGB与depth 帧数对应

        while not depth_frame or not color_frame:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.aligned_frames.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        self.dist_coeff = np.zeros([5,1])

        color_cam_intri = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        distortion = color_frame.get_profile().as_video_stream_profile().intrinsics.coeffs # for getting Distortion coefficient
        self.dist_coeff[0][0] = distortion[0]
        self.dist_coeff[1][0] = distortion[1]
        self.dist_coeff[2][0] = distortion[2]
        self.dist_coeff[3][0] = distortion[3]
        self.dist_coeff[4][0] = distortion[4]

        # depth_cam_intri = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()  # for ptc
        r = R.from_quat(cam_rotation)
        cam_rotation = r.as_matrix()
        self.cam_pose = np.zeros((4, 4), dtype=np.float32)
        self.cam_pose[:3, :3] = cam_rotation
        self.cam_pose[:3, 3:] = cam_translation
        self.cam_pose[-1, -1] = 1

        self.cam_intri = np.zeros([3, 3])
        # self.depth_cam_intri = np.zeros([3, 3])

        self.cam_intri[0][0] = color_cam_intri.fx
        self.cam_intri[1][1] = color_cam_intri.fy
        self.cam_intri[0][2] = color_cam_intri.ppx
        self.cam_intri[1][2] = color_cam_intri.ppy
        self.cam_intri[2][2] = 1

        self._cam_name = "RealSense"
        self.post_process = post_process

    def get_data(self):
        # 获取一帧图像
        frames = self.pipeline.wait_for_frames()
        # 将图像对齐
        aligned_frames = self.aligned_frames.process(frames)
        # 获取对齐后的深度图
        depth_frame = aligned_frames.get_depth_frame()
        # 获取对齐后的RGB图
        color_frame = aligned_frames.get_color_frame()

        while not depth_frame or not color_frame:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.aligned_frames.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

        depth_frame = self._post_process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def _post_process(self, frame):
        assert isinstance(frame, rs.frame)
        if isinstance(self.post_process, Iterable): # filter: 迭代器
            tmp_frame = frame.copy()
            for f in self.post_process: #判断自己的过滤器是否与realsense的过滤器同类
                # assert isinstance(f, rs.filter), error("Each element in post_process should be 'rs.filter'")
                assert isinstance(f, rs.filter)
                tmp_frame = f.process(tmp_frame)
            return tmp_frame
        elif isinstance(self.post_process, rs.filter):
            return self.post_process.process(frame)
        elif self.post_process is None:
            # Do nothing
            return frame
        else:
            # error("post_process should be one of '[rs.filter, ...]', 'rs.filter', or None")
            print("post_process should be one of '[rs.filter, ...]', 'rs.filter', or None")
            raise TypeError

    def close(self):
        self.pipeline.stop()

    def get_camera_params(self):
        return {
            "image_type": "bgr",
            "init_params": self.init_params,
            "depth_scale": self.depth_scale,
            "cam_intri": self.cam_intri.copy(),
            "cam_pose": self.cam_pose.copy(),
            "dist_coeff": self.dist_coeff.copy(),
            "cam_name": self._cam_name
        }

    def auto_calibration(self):
        pass
