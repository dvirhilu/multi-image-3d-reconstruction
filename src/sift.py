import numpy as np
import cv2
from utils.plt_utils import show_images
import matplotlib.pyplot as plt
from utils import file_io_utils

def draw_key_points(image, detailed=True, savefile=None):
    sift = cv2.SIFT_create()
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    images = file_io_utils.load_object_images("gluestick")

    images = images[:4]

    kp_images = [
        draw_key_points(image, detailed=False)
        for image in images
    ]

    show_images(*images)

    plt.show()
