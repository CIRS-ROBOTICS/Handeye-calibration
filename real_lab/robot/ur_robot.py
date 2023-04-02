import numpy as np

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from scipy.spatial.transform.rotation import Rotation as R

from ..end_effector.robotiq_gripper_control import RobotiqGripper
from ..end_effector.robotiq_usb_ctrl import RobotiqUSBCtrlGripper
from ..end_effector import robotiq_gripper
from .robot import Robot

from ..perception.vision.realsense import RealSense


class URRobot(Robot):
    def __init__(self, workspace_limits, robot_ip, rtu_gripper=False, home_joint_position=None):
        self.robot_ip = robot_ip
        self.rtu_gripper = rtu_gripper

        self.rtde_c = RTDEControl(self.robot_ip)
        self.rtde_r = RTDEReceive(self.robot_ip)

        if rtu_gripper:  # independent gripper control by USB
            self.gripper = RobotiqUSBCtrlGripper('/dev/ttyUSB0')
            self.gripper.activate()
            self.gripper.send_commond()
            self.gripper.reset()
            self.gripper.send_commond()
        else:
            self.gripper = robotiq_gripper.RobotiqGripper() # Creating gripper
            self.gripper.connect(self.robot_ip, 63352)      # Connecting to gripper
            self.gripper.activate()                         # Activating gripper after closing the gripper
            
            # # another method to directly control the gripper
            # self.gripper = RobotiqGripper(self.rtde_c)
            # self.gripper.set_speed(100)
            # self.gripper.activate()

        # Default joint speed configuration
        self._default_joint_vel = 1.05
        self._default_joint_acc = 1.4
        self._default_joint_tolerance = 0.01

        # Default tool speed configuration
        self._default_tool_vel = 0.3
        self._default_tool_acc = 1.2
        self._default_tool_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # set up camera
        self.camera = RealSense()
        self.cam_intrinsics = self.camera.get_camera_params()['cam_intri'] # get the camera intrinsic params

        if home_joint_position is None:
            self._home_joint_position = [88.41 * np.pi / 180, -128.30 * np.pi / 180, 126.03 * np.pi / 180,
                                         -87.80 * np.pi / 180, -89.46 * np.pi / 180, 138 * np.pi / 180]
        else:
            self._home_joint_position = home_joint_position

        self.reset()

    def get_observation(self):
        current_pose = self.rtde_r.getActualTCPPose()
        return current_pose[:3], current_pose[3:]

    def get_robot_params(self):
        return {
            "robot_ip": self.robot_ip,
            "is_rtu_gripper": self.rtu_gripper,
            "current_pose": self.rtde_r.getActualTCPPose()
        }

    def reset(self):
        self.move_joints(self._home_joint_position)
        # self.gripper.open()
        self.gripper.move_and_wait_for_pos(0, 255, 255) # set the suitable degree of gripper
        # first position: degree of opening gripper (adjust the size according to the calibration board)
        # second position: speed
        # third position: force

    def apply_action(self, *motor_commands, acc=None, vel=None):
        x, y, z, rx, ry, rz = motor_commands

        ori = [rx, ry, rz]
        r = R.from_euler('xyz', ori, degrees=False)
        orn = list(r.as_rotvec())
        target_pos = [x, y, z]

        if acc is None:
            acc = self._default_tcp_acc
        if vel is None:
            vel = self._default_joint_vel
        self.rtde_c.moveL(pose=target_pos + orn, acceleration=acc, speed=vel)

    def move_joints(self, joint_configuration, acc=None, vel=None):
        if acc is None:
            acc = self._default_joint_acc
        if vel is None:
            vel = self._default_joint_vel

        self.rtde_c.moveJ(q=joint_configuration, acceleration=acc, speed=vel)

    def close(self):
        self.rtde_c.stopScript()

    def go_home(self):
        self.move_joints(self._home_joint_position)

    def close_gripper(self):
        self.gripper.move_and_wait_for_pos(255, 255, 255) # (distance, speed, forece)

    def open_gripper(self):
        self.gripper.move_and_wait_for_pos(0, 255, 255) # (distance, speed, forece)

    def move_to(self, tool_position, tool_orientation, acc=None, vel=None): # move robot
        if acc is None:
            acc = self._default_tool_acc
        if vel is None:
            vel = self._default_tool_vel
        blend_1 = 0.0
        path = [[tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], vel, acc, blend_1]]
        self.rtde_c.moveL(path)

    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()

        return color_img, depth_img

