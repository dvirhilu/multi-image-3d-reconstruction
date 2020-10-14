'''
Multi-Image 3D Reconstruction
Automatic Camera Calibration

This script, given a set of ~20 images of a chessboard pattern at 
various angles, extracts the distortion parameters of the camera 
lens as well as the intrinsic parameters (focal length, principal 
point, skew coefficient).

The script takes in images from 
../camera_calibration/calib_images/<camera_name>/*
and outputs the camera parameters to 
../camera_calibration/calib_params/<camera_name>.yml

This script contains code from:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://medium.com/@aliyasineser/opencv-camera-calibration-e9a48bdd1844

Author:   Dvir Hilu
Date:     12/10/2020
'''

import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser(description='Camera Calibration Parameter Calculator')
parser.add_argument('-c', '--cameraname', default = "SamsungGalaxyA8", type = str,
        help = "The name of the camera used for calibration. Determines names of images" + 
                "the program will try to pull as well as the name of the file containing calibration data")
parser.add_argument('-s', '--square_size', default = 3, type = float,
        help = "The size of a square on the chess board in cm")
parser.add_argument('-w', '--width', default = 6, type = float,
        help = "Number of squares found on the thin side of the board")
parser.add_argument('-h', '--height', default = 9, type = float,
        help = "Number of squares found on the long side of the board")

args = parser.parse_args()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(camera_name, square_size, width, height):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob("../camera_calibration/" + camera_name + "/*")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, camera_name):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage("../camera_calibration/calib_params/" + camera_name + ".yml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_coefficients(camera_name):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage("../camera_calibration/calib_params/" + camera_name + ".yml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate(args.camera_name, args.square_size, args.width, args.height)
    save_coefficients(mtx, dist, args.camera_name)
    print("Calibration is finished. RMS: ", ret)