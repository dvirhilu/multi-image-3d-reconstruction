import glob
from os import getcwd
import cv2
import numpy as np
import yaml

def load_calib_images(calib_name):
    filenames = glob.glob("./images/calibration/" + calib_name + "/*")

    return [
        cv2.imread(file)
        for file in filenames
    ]

def load_object_images(object_name):
    filenames = glob.glob("./images/objects/" + object_name + "/*")

    print("Loading the following object images:", object_name)
    for filename in filenames:
        print(filename)

    return [
        cv2.imread(file)
        for file in filenames
    ]

def save_calib_coefficients(mtx, dist, calib_name):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage("./calibration_params/" + calib_name + ".yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("k", mtx)
    cv_file.write("d", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_calib_coefficients(calib_name):
    """ Loads camera matrix and distortion coefficients. """
    # set calibration file path
    filename = "./camera_calibration/calib_params/" + calib_name + ".yaml"

    with open(filename) as f:
        internal_params = yaml.load(f.read(), Loader=yaml.SafeLoader)

    k = np.array(internal_params["k"]["data"]).reshape(internal_params["k"]["rows"], internal_params["k"]["cols"])
    d = np.array(internal_params["d"]["data"]).reshape(internal_params["d"]["rows"], internal_params["d"]["cols"])
    
    return [k, d]