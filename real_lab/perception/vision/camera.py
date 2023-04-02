import numpy as np

from ..perception import Perception


class Camera(Perception):
    def get_data(self):
        """

        :return: Color Image with BGR style, Depth Image with Meters
        """
        raise NotImplementedError

    def close(self):
        """

        :return:
        """
        raise NotImplementedError

    def get_camera_params(self):
        """

        :return:
        """
        raise NotImplementedError

    def convert_ic_to_wc(self, image_coordinate, corresponding_depth):
        """

        :param image_coordinate: list of image coordinate of special point
        :param corresponding_depth: depth of the special point
        :param cam_pose: camera posture
        :param cam_intri:
        :return: corresponding world coordinate
        """
        cam_pose = self.cam_pose.copy()
        cam_pose = np.mat(cam_pose)

        cam_intri = np.mat(self.cam_intri.copy())

        assert len(image_coordinate) == 2
        image_coordinate.append(1)
        ic = np.array(image_coordinate)

        ic = ic * corresponding_depth
        ic = np.mat(ic)
        wc = np.dot(cam_intri.I, ic.T)

        obj2cam = np.vstack((wc, np.float64([1])))
        obj2base = np.dot(cam_pose, obj2cam)
        obj2base = np.delete(obj2base, 3, axis=0)
        obj2base = obj2base.T

        x, y, z = obj2base.item(0, 0), obj2base.item(0, 1), obj2base.item(0, 2)

        return x, y, z

    def get_pointcloud(self, color_img, depth_img):
        camera_intrinsics = self.cam_intri.copy()

        # Get depth image size
        im_h = depth_img.shape[0]
        im_w = depth_img.shape[1]

        # Project depth into 3D point cloud in camera coordinates
        pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
        cam_pts_x = np.multiply(pix_x - camera_intrinsics[0][2], depth_img / camera_intrinsics[0][0])
        cam_pts_y = np.multiply(pix_y - camera_intrinsics[1][2], depth_img / camera_intrinsics[1][1])
        cam_pts_z = depth_img.copy()
        cam_pts_x.shape = (im_h * im_w, 1)
        cam_pts_y.shape = (im_h * im_w, 1)
        cam_pts_z.shape = (im_h * im_w, 1)

        # Reshape image into colors for 3D point cloud
        rgb_pts_r = color_img[:, :, 0]
        rgb_pts_g = color_img[:, :, 1]
        rgb_pts_b = color_img[:, :, 2]
        rgb_pts_r.shape = (im_h * im_w, 1)
        rgb_pts_g.shape = (im_h * im_w, 1)
        rgb_pts_b.shape = (im_h * im_w, 1)

        cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
        rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

        return cam_pts, rgb_pts

    def get_heightmap(self, color_img, depth_img, workspace_limits, heightmap_resolution):

        cam_pose = self.cam_pose.copy()

        # Compute heightmap size
        heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                                   (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(
            int)

        # Get 3D point cloud from RGB-D images
        surface_pts, color_pts = self.get_pointcloud(color_img, depth_img)

        # Transform 3D point cloud from camera coordinates to robot coordinates
        surface_pts = np.transpose(np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) +  # Rotation
                                   np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0])))   # Translation

        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:, 2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = color_pts[sort_z_ind]

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
            np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
            surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
                                             surface_pts[:, 2] < workspace_limits[2][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]

        # Create orthographic top-down-view RGB-D heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
        color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
        color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
        color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
        z_bottom = workspace_limits[2][0]
        depth_heightmap = depth_heightmap - z_bottom
        depth_heightmap[depth_heightmap < 0] = 0
        depth_heightmap[depth_heightmap == -z_bottom] = np.nan

        return color_heightmap, depth_heightmap
