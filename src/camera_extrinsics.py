import numpy as np
import cv2
import filter_masks
import utils.file_io_utils as io
import utils.plt_utils as plt_utils
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def find_chessboard_corners(image, use_cv=True, display=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = img_gray.shape[:2]
    
    if use_cv:

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img_gray, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print("corners found!")
            # objpoints.append(objp)

            # find sub-pixel location of corners
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            # imgpoints.append(corners2)

            # Draw and display the corners
            if display:
                corner_image = cv2.drawChessboardCorners(image, (rows, cols), corners, ret)
                plt_utils.show_image(corner_image, title="Image with Chessboard Corners Highlighted")
        else:
            print("corners not found!", corners)
    else:
        raise RuntimeError("Not implemented yet")

if __name__=="__main__":
    images = io.load_object_images("monkey_thing")
    print(len(images))
    image = images[0]
    find_chessboard_corners(image, display=True)
    plt.show()

