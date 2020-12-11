import glob
from os import getcwd
import cv2
import numpy as np
import yaml

def load_object_images(object_name):
    '''
    @brief  Loads a set of object images based on the object name

    @param objecc_name The name of the directory in images/objects containing 
                        the object images
    @return             A list of loaded images
    '''
    filenames = glob.glob("./images/objects/" + object_name + "/*")

    print("Loading the following object images:", object_name)
    for filename in filenames:
        print(filename)

    return [
        cv2.imread(file)
        for file in filenames
    ]

def load_calib_coefficients(calib_name):
    '''
    @brief  Loads a set of camera calibration coefficients based on calib_name

    For more info:
    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    @param calib_name   The name of the directory in 
                        camera_calibration/calib_params containing the
                        calibration parameters
    @return k           The 3x3 intrinsic calibration matrix
    @return d           The (5,) shape ndarray containing distrotion
                        parameters in the form [k1, k2, p1, p2, p3]
    '''
    # set calibration file path
    filename = "./camera_calibration/calib_params/" + calib_name + ".yaml"

    with open(filename) as f:
        internal_params = yaml.load(f.read(), Loader=yaml.SafeLoader)

    k = np.array(internal_params["k"]["data"]).reshape(internal_params["k"]["rows"], internal_params["k"]["cols"])
    d = np.array(internal_params["d"]["data"]).reshape(internal_params["d"]["rows"], internal_params["d"]["cols"])
    
    return [k, d]