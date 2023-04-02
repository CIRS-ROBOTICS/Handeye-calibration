import copy
import numpy as np
from real_lab.end_effector.robotiq_gripper_control import RobotiqGripper
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


class Robot():
    def __init__(self, ip='192.168.1.102'):
        self.rob_c = RTDEControl(ip)
        self.rob_r = RTDEReceive(ip)
        self.robotiqGrip = RobotiqGripper(self.rob_c)

    # For UR robot
    def GoHome(self, speed=0.1, acceleration=0.1, verbose=False):
        home_pose = [-0.274, -0.353, 0.229, 3.14, 0, 0]
        if verbose:
            print('Go Home:{}'.format(str(home_pose)))
        self.rob_c.moveL(home_pose, speed, acceleration)

    def Pick(self, pose, speed=0.1, acceleration=0.1, verbose=False):
        pre_height = 0.1
        if verbose:
            print('pick:{}'.format(str(pose)))

        # open gripper
        self.robotiqGrip.open()

        # pre_pick_pos
        pre_pick_pose = copy.deepcopy(pose)
        pre_pick_pose[2] = pre_pick_pose[2] + pre_height
        self.rob_c.moveL(pre_pick_pose, speed, acceleration)

        # pick_pos and close gripper
        pick_pose = copy.deepcopy(pose)
        self.rob_c.moveL(pick_pose, speed, acceleration)

        self.robotiqGrip.close()
        # time.sleep(0.5)

        # post pick_pos
        post_pick_pos = pre_pick_pose
        self.rob_c.moveL(post_pick_pos, speed, acceleration)

    def Place(self, pose, speed=0.1, acceleration=0.1, verbose=False):
        pre_height = 0.1
        if verbose:
            print('place:{}'.format(str(pose)))

        # pre_place
        pre_place_pose = copy.deepcopy(pose)
        pre_place_pose[2] = pre_place_pose[2] + pre_height
        self.rob_c.moveL(pre_place_pose, speed, acceleration)

        # place and open gripper
        piace_pose = copy.deepcopy(pose)
        self.rob_c.moveL(piace_pose, speed, acceleration)
        self.robotiqGrip.open()
        # time.sleep(0.5)

        # post place
        post_place_pos = pre_place_pose
        self.rob_c.moveL(post_place_pos, speed, acceleration)

        # go home
        self.GoHome(speed, acceleration)

    def Joint2L(self, angle, speed=0.1, acceleration=0.1):
        # this is design to calculate the sixth axis angle when performing rope manipulation.
        pass

    def NotReach(self, tar, threshold=0.00001):
        '''
        focus on position
        Args:
            tar:
            threshold:

        Returns:

        '''
        cur = np.array(self.rob_r.getActualTCPPose()[:3])
        if np.linalg.norm(cur - tar) < threshold:
            return False
        else:
            return True

if __name__ == "__main__":
    robot = Robot()
    robot.GoHome()
    tar = robot.rob_r.getActualTCPPose()

    tar[2] -= 0.1
    # print('target: {}'.format(str(tar)))
    #
    # # while True:
    # #     print(ft.GetValue())
    #
    robot.rob_c.moveL(tar, 0.05, 0.05, async=True)
