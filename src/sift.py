import numpy as np
import cv2
import utils.plt_utils as plt_utils
import matplotlib.pyplot as plt
import utils.file_io_utils as io
from process_image_background import undistort, get_undistored_k_matrix

def draw_key_points(image, detailed=True, savefile=None):
    sift = cv2.SIFT_create()

    if len(image.shape) > 2:
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        im_gray = image

    kp = sift.detect(image,None)
    
    if detailed:
        im_kp=cv2.drawKeypoints(im_gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        im_kp=cv2.drawKeypoints(im_gray, kp, image)

    if savefile:
        cv2.imwrite(savefile, im_kp)
    
    return im_kp

def get_keypoints(image):
    sift = cv2.SIFT_create()
    return sift.detect(image, None)

def get_key_point_descriptors():
    pass

if __name__=="__main__":
    camera_calib = "SamsungGalaxyA8"
    
    # first, grab camera and distortion matrices
    k, d = io.load_calib_coefficients(camera_calib)

    # generate image points
    images = io.load_object_images("monkey_thing")
    good_indices = [0, 2, 4, 5, 7, 10, 11]
    # good_indices = [0, 2]
    
    images = [
        images[i] 
        for i in good_indices
    ]
    
    plt_utils.show_images(*images)

    #########################
    # undistort images
    #########################
    undistort_tuples = [
        get_undistored_k_matrix(image, k, d)
        for image in images
    ]

    k_mats, rois = zip(*undistort_tuples)

    images = [
        undistort(image, k, d, k_adj, roi)
        for (image, k_adj, roi) in zip(images, k_mats, rois)
    ]

    im_gray = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]

    plt_utils.show_images(*images)
    plt_utils.show_images(*im_gray)

    kp_images = [
        draw_key_points(image, detailed=False)
        for image in images
    ]

    plt_utils.show_images(*images)

    plt.show()
