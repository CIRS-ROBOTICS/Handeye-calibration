import numpy as np
import cv2
import sys
from cv2 import aruco 
from real_lab.perception.vision.realsense import RealSense
import math
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R) # get the rotation matrix
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    """
    pitch = atan2( -r20, sqrt(r00*r00+r10*r10) );
    yaw   = atan2(  r10, r00 );
    roll  = atan2(  r21, r22 );
    """
    if not singular: # rad
        x = math.atan2(R[2, 1], R[2, 2]) # roll: rotation about the x-axis
        y = math.atan2(-R[2, 0], sy) # pitch: rotation about the y-axis
        z = math.atan2(R[1, 0], R[0, 0]) # yaw: rotation around the z-axis
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # rad->deg
    rx = np.rad2deg(x)
    ry = np.rad2deg(y)
    rz = np.rad2deg(z)
    return rx, ry, rz
def get_gripper2base():
    robot_ip = '192.168.1.109'
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip) # get the robot information

    pose = rtde_r.getActualTCPPose()
    mat = R.from_rotvec(pose[3:]).as_matrix() # rotation matrix of gripper to base 3*3
    trans = pose[:3]
    trans = np.expand_dims(trans,axis=0)
    T_g2b = np.concatenate((mat, trans.T),axis=1)
    T_g2b = np.concatenate((T_g2b, np.array([[0,0,0,1]])),axis=0)

    return T_g2b


aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL) # generate the aruco dict
# The website of generating aruco marker is https://chev.me/arucogen/
marker_id = 25 
arucoParams = aruco.DetectorParameters_create()

# marker_img = aruco.drawMarker(aruco_dict, marker_id, 500)
# cv2.imshow("marker",marker_img)
# key = cv2.waitKey(1000)
# pic_name = "marker" + str(marker_id) + ".jpg"
# cv2.imwrite(pic_name,marker_img)

# camaera init
cam = RealSense()
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
rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, K, dist)
cv2.aruco.drawDetectedMarkers(rgb, corners, ids)
cv2.drawFrameAxes(rgb, K, dist, rvec, tvec, 0.03)
cv2.imshow("marker",rgb)
key = cv2.waitKey(3000)

# # poses visualization
# for i in range(rvec.shape[0]):
#     cv2.drawFrameAxes(rgb, K, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
#     aruco.drawDetectedMarkers(rgb, corners)
# # draw ID #
# cv2.putText(rgb, "Id: " + str(ids), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
# EulerAngles = rotationVectorToEulerAngles(rvec)
# EulerAngles = [round(i, 2) for i in EulerAngles]
# cv2.putText(rgb, "Attitude_angle:" + str(EulerAngles), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
#             cv2.LINE_AA)
# tvec = tvec * 1000
# for i in range(3):
#     tvec[0][0][i] = round(tvec[0][0][i], 1)
# tvec = np.squeeze(tvec)
# cv2.putText(rgb, "Position_coordinates:" + str(tvec) + str('mm'), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
#             cv2.LINE_AA)

# cv2.imshow("frame", rgb)
# key = cv2.waitKey(1000)

R_matrix= np.zeros((3, 3), dtype=np.float64)
cv2.Rodrigues(rvec, R_matrix) # get the rotation matrix
tvec = np.squeeze(tvec, axis=1)
tvec = tvec
T_t2c = np.hstack((R_matrix, tvec.T)) # marker2camera
print("\t\tThe transformation matrix of Aruco Board in camera coordinate system:\n", T_t2c)
T_t2c = np.concatenate((T_t2c, np.array([[0,0,0,1]])))
tvec = np.append(tvec, [1]) # the aruco position
tvec = np.expand_dims(tvec, axis=0)
print("\t\tThe position of Aruco Board in camera coordinate system:\n", tvec)

T_c2g = np.loadtxt('./result/eye_in_hand/real_param/camera2gripper.txt')

tvec_g = np.dot(T_c2g, tvec.T)
print("\t\tThe position of Aruco Board in gripper coordinate system:\n", tvec_g)

T_t2g = np.dot(T_c2g, T_t2c) # marker2gripper
print("\t\tThe transformation matrix of Aruco Board in gripper coordinate system:\n", T_t2g)

T_g2b = get_gripper2base() # gripper2base
print(T_g2b)
T_t2b = np.dot(T_g2b ,T_t2g) 
print("\t\tThe transformation matrix of Aruco Board in base coordinate system:\n", T_t2b)

# T_tcp = T_t2b

R_marker = np.array([[-1, 0, 0],
                     [0, 1, 0],
                     [0, 0, -1],
                     [0, 0, 0]]) # Rotate 180 degrees along the y-axis of the marker coordinate system
t_marker = np.array([[0, 0, 0, 1]])  
T_marker = np.concatenate((R_marker, t_marker.T), axis=1) 


T_tcp = np.dot(T_t2b, T_marker) # The marker pose is converted into a grasping pose
print("\t\tThe pose of TCP:\n", T_tcp)

# error correction and real robot test
control_pose = np.zeros(6)
control_pose[0] = T_tcp[0][3] + 0.002
control_pose[1] = T_tcp[1][3] - 0.003
control_pose[2] = T_tcp[2][3] - 0
control_ori_mat = np.zeros((3,3))
control_ori_mat = T_tcp[:3,:3]
control_ori_vec, _ = cv2.Rodrigues(control_ori_mat)
control_pose[3:] = control_ori_vec.squeeze()
print("\t\tThe rotation vector for TCP:\n", control_pose)

control_pose[2] += 0.1
path = np.append(control_pose ,np.array([0.2, 0.1, 0]))
path = [path]
rtde_c = rtde_control.RTDEControlInterface('192.168.1.109') # control the robot
print(rtde_c.moveL(path))
path[0][2] -= 0.1
print(rtde_c.moveL(path))