import glob
from os import getcwd
import cv2

def load_calib_images(calib_name):
    cwd = getcwd()
    filenames = glob.glob(cwd + "/images/calibration/" + calib_name + "/*")

    return [
        cv2.imread(file)
        for file in filenames
    ]

def load_object_images(object_name):
    cwd = getcwd()
    filenames = glob.glob(cwd + "/images/objects/" + object_name + "/*")

    return [
        cv2.imread(file)
        for file in filenames
    ]

def save_calib_coefficients(mtx, dist, calib_name):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cwd = getcwd()
    
    cv_file = cv2.FileStorage(cwd + "/calibration_params/" + calib_name + ".yml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("k", mtx)
    cv_file.write("d", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()

def load_calib_coefficients(calib_name):
    """ Loads camera matrix and distortion coefficients. """
    cwd = getcwd()
    
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(cwd + "/camera_calibration/calib_params/" + calib_name + ".yml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("k").mat()
    dist_matrix = cv_file.getNode("d").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]