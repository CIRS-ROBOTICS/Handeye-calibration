import numpy as np
import cv2
import sys
import rtde_control
import rtde_receive
from cv2 import aruco 
from real_lab.perception.vision.realsense import RealSense
import math
from scipy.spatial.transform import Rotation as R
import time
from utils import *

'''
User setting params
'''
marker_id = 25
maker_size = 0.06
robot_ip = "10.5.14.112"
tool_offset = [0.0, 0.0, 0.2115, 0.0, 0.0, 0.0] # tool offset when gripper closed
user_tool_offset = [0.0, 0.0, 0.18, 0.0, 0.0, 0.0] # user set tool offset when gripper opened

################################################################################################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL) # generate the aruco dict
# The website of generating aruco marker is https://chev.me/arucogen/
arucoParams = aruco.DetectorParameters_create()

# marker_img = aruco.drawMarker(aruco_dict, marker_id, 500)
# cv2.imshow("marker",marker_img)
# key = cv2.waitKey(1000)
# pic_name = "marker" + str(marker_id) + ".jpg"
# cv2.imwrite(pic_name,marker_img)

# camaera init
cam = RealSense()
time.sleep(1)
rgb, depth = cam.get_data()
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# detect marker
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict,parameters=arucoParams)
if len(corners) <= 0:
    print("Error: the corner is not detected")
    sys.exit()

# get the camera parameters
K = cam.get_camera_params()['cam_intri'] # camera intrisic
dist = cam.get_camera_params()['dist_coeff']
# estimate the pose of marker

# Note：let the rotation matrix of aruco coordinate system aligns with that of robot base coordinate system. 
rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, maker_size, K, dist)
cv2.aruco.drawDetectedMarkers(rgb, corners, ids)
cv2.drawFrameAxes(rgb, K, dist, rvec, tvec, 0.03)
cv2.imshow("marker",rgb)
key = cv2.waitKey(3000)
print("rvec, tvec",rvec, tvec)


R_matrix= np.zeros((3, 3), dtype=np.float64)
cv2.Rodrigues(rvec, R_matrix) # get the rotation matrix

# initialize the robot
rtde_c = rtde_control.RTDEControlInterface(robot_ip) # control the robot
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip) # get the robot information

# initialize the gripper
gripper = robotiq_gripper.RobotiqGripper() # Creating gripper
gripper.connect(robot_ip, 63352)    # Connecting to gripper
# gripper.activate()   

rtde_c.setTcp(user_tool_offset) # restore to the user tool coordinate system, we need to calibrate the relationships between camera and user designed tool coordinate system, not tool self coordinate system
# print("second set")
# print(rtde_c.getTCPOffset())
T_g2b_trans, T_g2b_mat = get_gripper2base(rtde_r) # gripper2base                      

checkboard_center = get_checkboard_position(rtde_c, rtde_r, gripper, tool_offset)
checkboard_center = np.array(checkboard_center).reshape(3, 1) # translation of aruco2base 


base2cam_trans = np.dot(R_matrix, -checkboard_center) + tvec.reshape(3,1) # base2cam_trans; -checkboard_center: translation of base2aruco

T_g2base = np.concatenate((T_g2b_mat, np.array(T_g2b_trans).reshape(3,1)), axis=1) # (3,4)
T_g2base = np.concatenate((T_g2base, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)

T_base2cam = np.concatenate((R_matrix, base2cam_trans), axis=1) # R_aruco2cam = R_base2cam 
T_base2cam = np.concatenate((T_base2cam, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0)

T_g2cam = np.matmul(T_base2cam, T_g2base)
np.savetxt("./result/eye_in_hand/real_param/init_g2cam.txt", T_g2cam)