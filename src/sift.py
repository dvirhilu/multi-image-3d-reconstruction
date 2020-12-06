import numpy as np
import cv2
import utils.plt_utils as plt_utils
import matplotlib.pyplot as plt
import utils.file_io_utils as io
from process_image_background import undistort, get_undistored_k_matrix, get_ordered_image_points
import utils.linalg_utils as linalg
from collections import namedtuple

PRINT = True

class SIFTFeature:
    def __init__(self, image_idx, coordinates, descriptor):
        self.image_idx = image_idx
        self.coordinates = coordinates
        self.descriptor = descriptor

Window = namedtuple("Window", "xmin xmax ymin ymax")

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

def find_window_of_interest(image_points):
    x_vals, y_vals = zip(*image_points)
    
    # coordinates of inner corners
    x_vals = x_vals[2:16:4]
    y_vals = y_vals[2:16:4]

    print(y_vals)

    return Window(min(x_vals), max(x_vals), min(y_vals), max(y_vals))

def kp_in_bound(kp, window):
    coords = kp.pt

    x_in_range = window.xmin < coords[0] < window.xmax
    y_in_range = window.ymin < coords[1] < window.ymax

    return x_in_range and y_in_range

def get_sift_feature_objects(image, image_index, window):
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(image, None)

    print(len(kps))
    print(image_index)

    feature_list =  [
        SIFTFeature(image_index, kp.pt, des)
        for (kp, des) in zip(kps, descs)
        if kp_in_bound(kp, window)
    ]

    print(len(feature_list))
    return feature_list

def find_match(feature, feature_list, ratio_threshold=0.7):
    for potential_match in feature_list:
        if potential_match.image_idx != feature_list[0].image_idx:
            raise ValueError("Features must be from the same image")

    # copy list to not change it
    potential_matches = feature_list[:]

    # sort index list based on distance
    potential_matches.sort(
        key=lambda potential_match: linalg.get_euclidean_distance(feature.descriptor, potential_match.descriptor)
    )

    best, second_best = potential_matches[:2]
    
    # check if match is valid
    min_dist = linalg.get_euclidean_distance(best.descriptor, feature.descriptor)
    next_dist = linalg.get_euclidean_distance(second_best.descriptor, feature.descriptor)
    valid_match = min_dist / next_dist < ratio_threshold

    global PRINT
    if PRINT:
        print(feature.image_idx, best.image_idx)
        print(best.coordinates)
        print(second_best.coordinates)
        print(feature.coordinates)
        print(min_dist, next_dist, valid_match)
        PRINT = False

    return (valid_match, best)

def find_match_group(feature, feature_lists, ratio_threshold=0.7):
    match_tuples = [
        find_match(feature, feature_list, ratio_threshold=ratio_threshold)
        for feature_list in feature_lists
        if len(feature_list) > 1
    ]

    # add current feature as a valid feature in the group
    match_tuples.append((True, feature))

    return tuple(
        match_tuple[1]
        for match_tuple in match_tuples
        if match_tuple[0]
    )

def group_feature_matches(feature_lists, ratio_threshold=0.7):
    feature_groups = []
    num_lists = len(feature_lists)
    for i in range(num_lists):
        for feature in feature_lists[i]:
            # delete all features in the group from their feature lists
            # this is done in order to not repeat groups
            # skip last element since it is the current feature
            already_matched = any(
                feature in group
                for group in feature_groups
            )

            if already_matched:
                continue

            # first, get all feature lists except current one
            other_lists = [
                feature_lists[j]
                for j in range(num_lists)
                if j != i
            ]

            # get all matches to current feature
            group = find_match_group(feature, other_lists, ratio_threshold=ratio_threshold)

            # skip group if it only contains current feature
            if len(group) == 1:
                continue


            # append group to group list
            feature_groups.append(group)

    return feature_groups


if __name__=="__main__":
    camera_calib = "SamsungGalaxyA8"
    
    # first, grab camera and distortion matrices
    k, d = io.load_calib_coefficients(camera_calib)

    # generate image points
    images = io.load_object_images("eraser")
    # good_indices = [0, 2, 4, 5, 7, 10, 11]
    # good_indices = [0, 2]
    # images = [
    #     images[i] 
    #     for i in good_indices
    # ]
    titles = [
        "Image %0d" %i
        for i in range(len(images))
    ]

    im_gray = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]
    plt_utils.show_images(*images)

    #########################
    # undistort images
    #########################
    undistort_tuples = [
        get_undistored_k_matrix(image, k, d)
        for image in im_gray
    ]

    k_mats, rois = zip(*undistort_tuples)

    print(k_mats)

    images = [
        undistort(image, k, d, k_adj, roi)
        for (image, k_adj, roi) in zip(im_gray, k_mats, rois)
    ]

    plt_utils.show_images(*im_gray)
    #########################
    # find image points
    #########################

    windowsize=10
    sobel_size=3
    harris_const=0.04
    harris_threshold=0.1
    r=40
    p=0.5
    d = 150
    ret_tuples = [
        get_ordered_image_points(image, windowsize=windowsize, sobel_size=sobel_size, k=harris_const, harris_threshold=harris_threshold, r=r, p=p, d=d)
        for image in im_gray
    ]

    is_valids, image_points, corners = zip(*ret_tuples)

    image_points = [
        points
        for (points, is_valid) in zip(image_points, is_valids)
        if is_valids
    ]
    corners = [
        corner
        for (corner, is_valid) in zip(corners, is_valids)
        if is_valid
    ]
    
    im_gray = [
        image
        for (image, is_valid) in zip(im_gray, is_valids)
        if is_valid
    ]

    plt_utils.plot_image_points(im_gray, image_points, titles=titles, sup_title="Image Chessboard Points")
    plt_utils.plot_point_path(im_gray, corners, image_points, titles=titles, sup_title="Corner Point Sequence")

    #########################
    # Match Key Features
    #########################

    # TODO: consider if to make this more constrainted and hard-coded for efficiency
    windows = [
        find_window_of_interest(points)
        for points in image_points
    ]

    print(windows)

    kp_images = [
        draw_key_points(image, detailed=False)
        for image in images[:]
    ]

    plt_utils.show_images(*kp_images, titles=titles, sup_title="SIFT Keypoints")

    # get features
    features = [
        get_sift_feature_objects(im_gray[i], i, window)
        for (i, window) in zip(range(len(im_gray)), windows)
    ]

    for feature_list in features:
        print(len(feature_list))

    feature_points = [
        [
            feature.coordinates
            for feature in feature_list
        ]
        for feature_list in features
    ]

    plt_utils.plot_image_points(im_gray, feature_points, sup_title="All Feature Points")

    feature_groups = group_feature_matches(features)

    print(len(feature_groups))

    print("lengths")
    for group in feature_groups[190:197]:
        print(len(group))
        print([feature.image_idx for feature in group])
        print([feature.coordinates for feature in group])

    point_list = []
    for i in range(len(im_gray)):
        points = []
        for group in feature_groups[190:200]:
            added = False
            for feature in group:
                if feature.image_idx == i:
                    points.append(feature.coordinates)
                    added = True
            if not added:
                points.append(None)
        print(len(points))
        point_list.append(points)

    good_indices = [2, 4, 11, 12]
    im_gray = [
        im_gray[i]
        for i in good_indices
    ]
    point_list = [
        point_list[i]
        for i in good_indices
    ]
    titles = [
        titles[i]
        for i in good_indices
    ]

    plt_utils.plot_image_points(im_gray, point_list, titles=titles, sup_title="Matched Features", same_colour=False)

    plt.show()