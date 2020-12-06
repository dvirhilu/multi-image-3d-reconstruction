import numpy as np
import cv2
import argparse
import utils.file_io_utils as io
from process_image_background import get_ordered_image_points, get_undistored_k_matrix, undistort
from camera_geometry import get_camera_extrinsic_matrix_nls
from sift import find_window_of_interest, get_sift_feature_objects, group_feature_matches
from reconstruct_3d_pos import reconstruct_3D_points, compute_reprojection_error_distribution

if __name__=="__main__":
    ##########################################################################
    # Parse command line arguments
    ##########################################################################
    parser = argparse.ArgumentParser(
        description='Multi-Image 3D Reconstruction'
    )
    parser.add_argument(
        '-c', '--camera_name', default = "SamsungGalaxyA8", type = str,
        help = "The name of the camera calibration to be used for the \
                reconstruction. The calibration name represents the camera \
                used to take the images"
    )
    parser.add_argument(
        '-o', '--object_name', default = "eraser", type = str, 
        help = "The name of the folder which includes the object pictures. \
                Folder name should be specific and easily associatable with \
                the object"
    )

    args = parser.parse_args()
    
    ##########################################################################
    # Define key parameters
    ##########################################################################
    calib_name              = args.camera_name
    object_name             = args.object_name
    harris_windowsize       = 10
    harris_sobel_size       = 3
    harris_const            = 0.04
    harris_threshold        = 0.1
    centrosym_r             = 40
    centrosym_threshold     = 0.5
    distance_threshold      = 150
    length                  = 9
    width                   = 6
    square_size             = 1.9
    sift_ratio_threshold    = 0.7

    ##########################################################################
    # Load images and calibration
    ##########################################################################
    images_orig = io.load_object_images(object_name)
    K, distortion_params = io.load_calib_coefficients(calib_name)

    ##########################################################################
    # Undistort images and update intrinsic calibration matrix
    ##########################################################################
    # adjust intrinsic camera matrix to account for distortion in images
    undistort_tuple = [
        get_undistored_k_matrix(image, K, distortion_params)
        for image in images_orig
    ]
    K_mats, ROIs = zip(*undistort_tuple)

    # undistort images
    images = [
        undistort(image, K, distortion_params, K_adj, ROI)
        for (image, K_adj, ROI) in zip(images_orig, K_mats, ROIs)
    ]

    ##########################################################################
    # Locate and sort checkered image points
    ##########################################################################
    # get grayscale converted images
    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images_orig
    ]

    # return a sorted list of image locations of the chessboard cross corners
    ret_tuple = [
        get_ordered_image_points(image, windowsize=harris_windowsize, 
                                 sobel_size=harris_sobel_size, k=harris_const,
                                 harris_threshold=harris_threshold, 
                                 r=centrosym_r, p=centrosym_threshold, 
                                 d=distance_threshold)
        for image in images
    ]
    is_valid, image_points = zip(*ret_tuple)
    image_points = list(image_points)

    # find and remove failed image indices
    failed_indices = [
        len(is_valid) - 1 - i
        for i in range(len(is_valid))
        if not is_valid[i]
    ]

    print(failed_indices)

    # filter out failed images
    for i in failed_indices:
        del image_points[i]
        del images[i]
        del images_orig[i]

    ##########################################################################
    # Calculate projection matrix
    ##########################################################################
    # compute extrinsic camera calibration matrix
    G_mats = [
        get_camera_extrinsic_matrix_nls(points, K, length=length, width=width, 
                                        square_size=square_size)
        for (points, K) in zip(image_points, K_mats)
    ]

    # projection matrix is the product of intrinsic and extrinsic parameters
    proj_mats = [
        K @ G
        for (K, G) in zip(K_mats, G_mats)
    ]

    ##########################################################################
    # Use SIFT to match features between images
    ##########################################################################
    # constrain the SIFT search region to be within chessboard x corners
    windows = [
        find_window_of_interest(points)
        for points in image_points
    ]

    # use SIFT to create SIFTFeature objects
    # The SIFTFeature class is defined in src/sift.py
    features = [
        get_sift_feature_objects(images[i], i, windows[i])
        for i in range(len(images))
    ]

    # group features that match across more than one image together
    feature_groups = group_feature_matches(features, ratio_threshold=sift_ratio_threshold)

    ##########################################################################
    # Triangulate point cloud from matching image features
    ##########################################################################
    # generate initial point cloud
    point_cloud = reconstruct_3D_points(feature_groups, proj_mats)

    # compute reprojection error
    reprojection_error = compute_reprojection_error_distribution(
        point_cloud, 
        feature_groups, 
        proj_mats
    )

    print(max(reprojection_error), np.mean(reprojection_error), len(reprojection_error))

    print(sum(i > 100 for i in reprojection_error))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(reprojection_error, bins=1000)
    # plt.xlim((0, 1e5))
    plt.show()