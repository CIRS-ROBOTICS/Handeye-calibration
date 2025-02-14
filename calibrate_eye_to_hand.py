#!/usr/bin/env python
import numpy as np
import time
import cv2
import os
from utils import *
from scipy import optimize  
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_ip', type=str, default='10.5.13.66', help='Robot ip address')
    parser.add_argument('--calib_grid_step', type=float, default=0.05, help='Calibration grid step')
    parser.add_argument('--workspace', type=float, nargs=6, default=[-0.37, -0.11, -0.7, -0.55, 0.4, 0.6], 
                        help='Workspace range, [xmin, xmax, ymin, ymax, zmin, zmax]')
    parser.add_argument('--home_joint_position', type=float, nargs=6, 
                        default=[45.29, -88.62, 109.17, -198.91, -62.49, 270.17],
                        help='Robot arm joint angles at home pose, in degrees')
    parser.add_argument('--use_recorded_data', action='store_true', default=False, help='Use data collected before')
    parser.add_argument('--camera', type=str, default='default', choices=['L515', 'SR300', 'default'], help='Camera model')
    parser.add_argument('--checkboard_size', type=int, default=5, help='Calibration size')
    parser.add_argument('--user_tool_offset', type=float, nargs=6, default=[0.0, 0.0, 0.18, 0.0, 0.0, 0.0], help='user set tool offset relative to wrist when gripper opened')
    args = parser.parse_args()
    args.workspace = np.array(args.workspace).reshape(3, 2)
    args.home_joint_position = [np.deg2rad(x) for x in args.home_joint_position]
    return args


if __name__ == '__main__':
    args = parse_args()
    measured_pts = []
    observed_pts = []
    observed_pix = []

    calibration_dir = "./result/eye_to_hand/calibrate_pictures"
    calibration_param_dir = "./result/eye_to_hand/real_param"
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(calibration_param_dir, exist_ok=True)

    if not args.use_recorded_data:
        # --------------- Setup options ---------------
        home_joint_config = args.home_joint_position

        calib_grid_pts = build_workspace_points(args.workspace, args.calib_grid_step)
        num_z = calib_grid_pts.shape[1]

        rtde_c, rtde_r, gripper, camera = init_robot(args.robot_ip, args.camera)
        cam_intrinsics = camera.get_camera_params()['cam_intri'] # get the camera intrinsic params

        # setup robot parameters of robots
        vel = 0.2 # velocity
        acc = 0.1 # accelaration
        blend_1 = 0.0
        
        # Move to home pose
        rtde_c.moveJ(home_joint_config, vel, acc)

        # Make robot gripper point upwards ***
        joint_radian = [45.29, -88.62, 109.17, -198.91, -62.49, 270.17]
        joint_radian = tuple([np.deg2rad(x) for x in joint_radian])
        rtde_c.moveJ(joint_radian, vel, acc)

        print('Please place the calibration board at the end of the robot arm')
        print('Press any key to make sure the calibration board is installed successfully')
        press_any_key()
        # gripper.move_and_wait_for_pos(255, 255, 255)

        rtde_c.setTcp(args.user_tool_offset) # set user tool offset of user tool coordinate system relative to the base coordinate system(robot base)

        tool_orientation = rtde_r.getActualTCPPose()[3:6]

        # Move robot to each calibration point in workspace
        print('Collecting data...')
        # for calib_pt_idx in range(num_calib_grid_pts):
        for calib_pt_xy_idx in range(calib_grid_pts.shape[0]):
            for calib_pt_z_idx in range(calib_grid_pts.shape[1]):
                tool_position = calib_grid_pts[calib_pt_xy_idx, calib_pt_z_idx,:]
                path = [[tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2],
                        vel, acc, blend_1]]
                try:
                    rtde_c.moveL(path)
                except TypeError:
                    print('Pose not reachable')
                    continue
                time.sleep(0.5)
                
                # Find checkerboard center
                checkerboard_size = (args.checkboard_size, args.checkboard_size)    # the size of calibration board
                # checkerboard_size is the size of corner not the calibrate board size
                refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                camera_color_img, camera_depth_img = camera.get_data()  # Get the RGB-D image, the unit of the depth map is meters
                # cv2.imshow('Current',camera_color_img)
                gray_data = cv2.cvtColor(camera_color_img, cv2.COLOR_BGR2GRAY)
                # detect checkerboard position
                checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
                # corners： marked out footprints
                if checkerboard_found:
                    print("Checkerboard found")
                    corners_refined = cv2.cornerSubPix(gray_data, corners, checkerboard_size, (-1,-1), refine_criteria)

                    # get center conrer index
                    mid_corner_ind = np.trunc((checkerboard_size[0] * checkerboard_size[1]) / 2)

                    # Get observed checkerboard center 3D point in camera space
                    checkerboard_pix = np.round(corners_refined[int(mid_corner_ind),0,:]).astype(int)
                    # The depth value has a large error in the black area, so the mean value of the depth of the white area
                    # in the 7*7 neighborhood of the center of the calibration board is selected as
                    # the depth of the center of the calibration board
                    # Here is the neighborhoods of the center pixel
                    block = camera_depth_img[checkerboard_pix[1] - 3:checkerboard_pix[1] + 3, checkerboard_pix[0] - 3:checkerboard_pix[0] + 3].copy()
                    gary_block = gray_data[checkerboard_pix[1] - 3:checkerboard_pix[1] + 3, checkerboard_pix[0] - 3:checkerboard_pix[0] + 3].copy()
                    block[gary_block < np.mean(gary_block)] = 0     # reomve noisy black pixels on the image
                    # The center of the calibration board in the camera coordinate system
                    checkerboard_z = np.sum(block) / (np.sum(block != 0) + 1e-16)
                    checkerboard_x = np.multiply(checkerboard_pix[0] - cam_intrinsics[0][2], checkerboard_z / cam_intrinsics[0][0])
                    checkerboard_y = np.multiply(checkerboard_pix[1] - cam_intrinsics[1][2], checkerboard_z / cam_intrinsics[1][1])
                    print('Checkboard position: %f, %f, %f' % (checkerboard_x, checkerboard_y, checkerboard_z))
                    if checkerboard_z == 0:
                        print('Checkboard found but depth invalid')
                        break

                    # the center of board in the camera coordinate system
                    observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
                    
                    cur_state = rtde_r.getActualTCPPose()
                    tool_position = np.array(cur_state[:3]).reshape(1,3)

                    measured_pts.append(tool_position)  # The position of the calibration board in the base coordinate system
                    observed_pix.append(checkerboard_pix)   # center pixel position of the calibration board

                    # Draw and display the corners
                    # vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
                    vis = cv2.drawChessboardCorners(camera_color_img, (1,1), corners_refined[int(mid_corner_ind),:,:], checkerboard_found)
                    print(checkerboard_pix)
                    vis = cv2.putText(vis, 'ID: %d' % (calib_pt_xy_idx * num_z + calib_pt_z_idx), 
                                      (checkerboard_pix[0] + 3, checkerboard_pix[1] - 2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    vis = cv2.putText(vis, 'Depth: %.0f mm' % (checkerboard_z * 1000), 
                                      (checkerboard_pix[0] + 3, checkerboard_pix[1] + 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    vis = cv2.putText(vis, 'x=%d, y=%d' % (checkerboard_pix[0], checkerboard_pix[1]), 
                                      (checkerboard_pix[0] + 3, checkerboard_pix[1] + 18), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imwrite(os.path.join(calibration_dir,'%06d.png' % len(measured_pts)), vis)
                    cv2.imshow('Calibration',vis)
                else:
                    print('No checkboard found')
                    # break
                cv2.waitKey(5)

        # Move robot back to home pose
        rtde_c.moveJ(home_joint_config, vel, acc)

        measured_pts = np.asarray(measured_pts).reshape(-1, 3)
        observed_pts = np.asarray(observed_pts)
        observed_pix = np.asarray(observed_pix)
        np.save(os.path.join(calibration_param_dir, 'measured_pts.npy'), measured_pts)
        np.save(os.path.join(calibration_param_dir, 'observed_pix.npy'), observed_pix)
        np.save(os.path.join(calibration_param_dir, 'observed_pts.npy'), observed_pts)
        np.save(os.path.join(calibration_param_dir, 'cam_intrinsics.npy'), cam_intrinsics)
    else:
        measured_pts = np.load(os.path.join(calibration_param_dir, 'measured_pts.npy'), allow_pickle=True)
        observed_pts = np.load(os.path.join(calibration_param_dir, 'observed_pts.npy'), allow_pickle=True)
        observed_pix = np.load(os.path.join(calibration_param_dir, 'observed_pix.npy'), allow_pickle=True)
        cam_intrinsics = np.load(os.path.join(calibration_param_dir, 'cam_intrinsics.npy'), allow_pickle=True)
    



# ICP calibration
points_camera = observed_pts 
points_base = measured_pts

pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(points_camera)
pcd_cam.paint_uniform_color([0, 1.0, 0])

pcd_rot = o3d.geometry.PointCloud()
pcd_rot.points = o3d.utility.Vector3dVector(points_base)
pcd_rot.paint_uniform_color([0, 0, 1.0])

threshold = 0.01
trans_init = np.loadtxt("./result/eye_to_hand/real_param/init_base2cam.txt")
# Trans_init is T_target2source i.e. the representation of target coordinates in source coordinates
# In other words, the coordinate system {source} can be translated and rotated to obtain the coordinate system {target}
draw_registration_result(pcd_rot, pcd_cam, trans_init) # source pds transforms to target pds.

evaluation = o3d.pipelines.registration.evaluate_registration(pcd_rot, pcd_cam, threshold, trans_init)
print("\t\tBefore calibration:\n", evaluation)

# global ICP
voxel_size = 0.05
pcd_rot2 = copy.deepcopy(pcd_rot)
pcd_rot2.transform(trans_init)
source_down, source_fpfh = preprocess_point_cloud(pcd_rot2, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(pcd_cam, voxel_size)
global_result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
draw_registration_result(pcd_rot2, pcd_cam, global_result.transformation) 

# calibrate
trans = point_to_point_icp(pcd_rot, pcd_cam, threshold, global_result.transformation @ trans_init)

# visualization of calibration results
draw_registration_result(pcd_cam, pcd_rot, trans)

np.savetxt(os.path.join(calibration_param_dir, 'camera2base.txt'), trans)
print('Done')
