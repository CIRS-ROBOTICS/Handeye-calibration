import sys

import cv2

from robot_tool.utils.logger import error
from robot_tool.perception.vision.mech_eye import MechEye
from robot_tool.perception.vision.realsense import RealSense
from robot_tool.perception.vision.zed import ZED


def main():
    # camera_type = sys.argv[1]
    # camera_init_params = sys.argv[2]

    camera_type = "realsense"
    camera_init_params = None

    camera_type = camera_type.upper()
    if camera_type == "MECH-EYE":
        cam = MechEye(camera_init_params)

    elif camera_type == "ZED":
        cam = ZED()

    elif camera_type == "REALSENSE":
        cam = RealSense()

    else:
        error("Camera type must be one of {MECH-EYE, ZED, REALSENSE}!")
        raise NotImplementedError

    while True:
        color, depth = cam.get_data()
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Color", color)
        cv2.imshow("Depth", depth_image)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cam.close()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
