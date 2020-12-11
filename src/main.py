import numpy as np
import cv2
import argparse
import utils.file_io_utils as io
import utils.plt_utils as plt
from process_image_background import (
    get_ordered_image_points, 
    get_undistored_k_matrix, 
    undistort
)
from camera_geometry import (
    get_camera_extrinsic_matrix_nls, 
    find_high_error_proj_mat_indices
)
from sift import (
    find_window_of_interest, 
    get_sift_feature_objects, 
    group_feature_matches    
) 
from reconstruct_3d_pos import (
    reconstruct_3D_points, 
    filter_reprojection_error, 
    filter_xyz_outliers,
    shift_points_to_centroid,
    add_background_surface
)
from utils.view3D_utils import view_point_cloud_interactively

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
        '-o', '--object_name', type = str, 
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
    mean_error_threshold    = 10
    max_error_threshold     = 20
    length                  = 9
    width                   = 6
    square_size             = 1.9
    sift_ratio_threshold    = 0.7
    num_xy_stdev            = 2
    z_percentile            = 80

    ##########################################################################
    # Load images and calibration
    ##########################################################################
    images_orig = io.load_object_images(object_name)
    K, distortion_params = io.load_calib_coefficients(calib_name)

    ##########################################################################
    # Undistort images and update intrinsic calibration matrix
    ##########################################################################
    # adjust intrinsic camera matrix to account for distortion in images
    print("\nRemoving image distortion...")
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

    print("\nLocating X-corners in image...")
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

    # find failed image indices
    failed_indices = [
        i
        for i in range(len(is_valid)-1, -1, -1)
        if not is_valid[i]
    ]

    print("X-corners could not be located in the following images:", failed_indices)
    print("Remving them...")

    # filter out failed images
    for i in failed_indices:
        del image_points[i]
        del images[i]
        del images_orig[i]


    ##########################################################################
    # Calculate projection matrix
    ##########################################################################
    print("\nRunning NLS computation to find extrinsic camera parameters...")
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

    # find failed projection matrices
    failed_indices = find_high_error_proj_mat_indices(
        image_points,
        proj_mats,
        mean_error_threshold=mean_error_threshold,
        max_error_threshold=max_error_threshold,
        length=length,
        width=width,
        square_size=square_size
    )

    print("Projection error too high for the following images:", failed_indices)
    print("Remving them...")


    # filter out failed projections matrices
    for i in failed_indices:
        del image_points[i]
        del images[i]
        del proj_mats[i]
        del images_orig[i]

    ##########################################################################
    # Use SIFT to match features between images
    ##########################################################################
    # constrain the SIFT search region to be within chessboard x corners
    windows = [
        find_window_of_interest(points)
        for points in image_points
    ]

    print("\nUsing SIFT to detect image features...")
    # use SIFT to create SIFTFeature objects
    # The SIFTFeature class is defined in src/sift.py
    features = [
        get_sift_feature_objects(images[i], i, windows[i])
        for i in range(len(images))
    ]
    print("Summary of features detected in each image:")
    for i in range(len(images)):
        print("Image %0d" %i, len(features[i]))

    print("\nFinding SIFT feature match groups across all images...")
    # group features that match across more than one image together
    feature_groups = group_feature_matches(features, ratio_threshold=sift_ratio_threshold)
    print("%0d feature groups found" % len(feature_groups))

    ##########################################################################
    # Triangulate point cloud from matching image features
    ##########################################################################
    print("\nTriangulating SIFT feature points...")
    # generate initial point cloud
    point_cloud = reconstruct_3D_points(feature_groups, proj_mats)
    
    # report number of points
    original_size = len(point_cloud)
    print("%0d point cloud generated!" % original_size)

    print("\nRemoving points with high reprojection error...")
    # filter points based on reprojection error
    point_cloud = filter_reprojection_error(
        point_cloud,
        feature_groups,
        proj_mats,
        reprojection_error_threshold=mean_error_threshold
    )
    
    # report number of rejected points
    new_size = len(point_cloud)
    print(
        "%0d points (%.2f %%) removed from point cloud due to " \
        "high reprojection error" % \
        ((original_size - new_size), 100*(1-new_size/original_size))
    )
    
    # filter points based on x,y,z distribution outliers
    point_cloud = filter_xyz_outliers(
        point_cloud,
        num_stdev=num_xy_stdev,
        z_percentile=z_percentile
    )
    
    # report number of rejected points
    original_size = new_size
    new_size = len(point_cloud)
    print(
        "%0d points (%.2f %%) removed from point cloud due to " \
        "being outliers in XYZ distributions" % \
        ((original_size - new_size), 100*(1-new_size/original_size))
    )

    # # add background surface
    # point_cloud = add_background_surface(point_cloud, num_stdev=num_xy_stdev)

    print("\nStarting up interactive point cloud viewer with %0d points..."\
          % len(point_cloud))
    # shift point cloud to new centroid
    point_cloud = shift_points_to_centroid(point_cloud)

    view_point_cloud_interactively(point_cloud)