import numpy as np
from real_lab.perception.vision.realsense import RealSense
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import termios
import os
import rtde_control
import rtde_receive
from real_lab.end_effector import robotiq_gripper
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R 


def build_workspace_points(workspace_limits, calib_grid_step):
    # Construct 3D calibration grid across workspace
    gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 
                              1 + round((workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
    gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 
                              1 + round((workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
    gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 
                              1 + round((workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step))
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z) 
    # Generate a grid point coordinate matrix, the input x, y, z is the horizontal and vertical coordinate column vector of the grid point (not matrix) The output X, Y, Z is the coordinate matrix
    # Put the first and second coordinates of the elements in the Cartesian product of the two arrays into two matrices respectively
    num_z = calib_grid_x.shape[2]
    calib_grid_x = calib_grid_x.reshape(-1, num_z, 1)
    calib_grid_y = calib_grid_y.reshape(-1, num_z, 1)
    calib_grid_z = calib_grid_z.reshape(-1, num_z, 1)
    calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=2)     # [num_x * num_y, num_z, 3]
    return calib_grid_pts


def init_robot(robot_ip, camera_model):
    # initialize the robot
    rtde_c = rtde_control.RTDEControlInterface(robot_ip) # control the robot
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip) # get the robot information

    # initialize the gripper
    gripper = robotiq_gripper.RobotiqGripper() # Creating gripper
    gripper.connect(robot_ip, 63352)    # Connecting to gripper
    # gripper.activate()                         

    # set up camera
    camera = RealSense(model=camera_model)
    return rtde_c, rtde_r, gripper, camera


def press_any_key():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd) 
    new_ttyinfo = old_ttyinfo[:] 
    new_ttyinfo[3] &= ~termios.ICANON 
    new_ttyinfo[3] &= ~termios.ECHO 
    sys.stdout.flush()
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo) 
    os.read(fd, 7) 
    termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo) 


def point_to_plane_icp(source, target, threshold, trans_init):
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation, "\n")
    print(np.linalg.inv(reg_p2l.transformation),'\n')
    #draw_registration_result(source, target, reg_p2l.transformation)


def point_to_point_icp(source, target, threshold, trans_init):
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print("\t\tAfter calibration:\n", reg_p2p)
    print("Transformation is:")
    # print(reg_p2p.transformation, "\n")
    print(np.linalg.inv(reg_p2p.transformation),'\n')
    
    return np.linalg.inv(reg_p2p.transformation)

    #draw_registration_result(source, target, reg_p2p.transformation)


def draw_registration_result(source, target, transformation):
    base = o3d.geometry.TriangleMesh.create_coordinate_frame()
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp,
         target_temp])
    
def get_checkboard_position(rtde_r, gripper):
    gripper.move_and_wait_for_pos(255, 255, 255) # close gripper
    
    print('Please drag the robot arm to the checkboard center position')
    print('Press any key to confirm the position')
    press_any_key()

    tool_offest = 0.0135 # NOTE: The deviation from the actual state on the z axis after the gripper is closed

    cur_state = rtde_r.getActualTCPPose()
    checkboard_center = cur_state[:3]
    checkboard_center[2] -= tool_offest # Correction of z-axis deviation

    print('Checkboard position: [%f, %f, %f]' % tuple(checkboard_center))
    return checkboard_center

def get_gripper2base(rtde_r):
    pose = rtde_r.getActualTCPPose()
    mat = R.from_rotvec(pose[3:]).as_matrix() # rotation matrix of gripper to base 3*3
    trans = pose[:3]
    # trans = np.expand_dims(trans,axis=0)
    # T_g2b = np.concatenate((mat, trans.T),axis=1)
    # T_g2b = np.concatenate((T_g2b, np.array([[0,0,0,1]])),axis=0)

    return trans, mat

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result