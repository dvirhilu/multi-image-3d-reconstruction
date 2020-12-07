import numpy as np
import utils.linalg_utils as linalg
import utils.file_io_utils as io
from process_image_background import get_ordered_image_points, undistort, get_undistored_k_matrix
from utils import plt_utils
import cv2
from sift import SIFTFeature
from camera_geometry import get_camera_extrinsic_matrix_nls, get_world_points
import matplotlib.pyplot as plt
from itertools import combinations

COUNT=0

def create_ls_matrix(proj_mats, image_points):
    x_cross_prod_eqns = np.array([
        point[0]*P[2, :].T - P[0, :].T
        for (point, P) in zip(image_points, proj_mats)
    ])

    y_cross_prod_eqns = np.array([
        point[1]*P[2, :].T - P[1, :].T
        for (point, P) in zip(image_points, proj_mats)
    ])

    return np.concatenate((x_cross_prod_eqns, y_cross_prod_eqns), axis=0)

def reconstruct_3D_points(feature_groups, proj_mats):
    reconstructed_points = []
    for group in feature_groups:
        relevant_proj_mats = [
            proj_mats[feature.image_idx]
            for feature in group
        ]

        image_points = [
            feature.coordinates
            for feature in group
        ]

        ls_mat = create_ls_matrix(relevant_proj_mats, image_points)

        world_point = linalg.solve_homogeneous_least_squares(ls_mat)

        # convert back from homogeneous coordinates
        world_point = world_point[0:3] / world_point[3]

        reconstructed_points.append(world_point)

    return reconstructed_points

def compute_mean_reprojection_error(reconstructed_point, feature_group, P_mats):
    global COUNT
    total_error = 0
    for feature in feature_group:
        # compute reprojection
        P = P_mats[feature.image_idx]
        X = np.append(reconstructed_point, [1], axis=0).reshape(4,1)
        reprojection_homogeneous = P @ X
        reprojection = reprojection_homogeneous[:2, 0] / reprojection_homogeneous[2,0]

        # get original image point
        image_point = np.array(feature.coordinates)

        total_error += linalg.get_euclidean_distance(reprojection, image_point)

    return total_error / len(feature_group)

def compute_reprojection_error_distribution(reconstructed_points, feature_groups, P_mats):
    return [
        compute_mean_reprojection_error(point, group, P_mats)
        for (point, group) in zip(reconstructed_points, feature_groups)
    ]

def filter_reprojection_error(reconstructed_points, feature_groups, P_mats, reprojection_error_threshold=20):
    # compute mean reprojection error for each reconstructed point
    reprojection_error = compute_reprojection_error_distribution(
        reconstructed_points, 
        feature_groups, 
        proj_mats
    )

    # filter out points above MAE threshold
    return [
        reconstructed_points[i]
        for i in range(len(reconstructed_points))
        if reprojection_error[i] < reprojection_error_threshold
    ]

def filter_xyz_outliers(reconstructed_points, num_stdev=1, z_percentile=80):
    # filter points with z < 0
    reconstructed_points = [
        point
        for point in reconstructed_points
        if point[2] >= 0
    ]

    # compute point centroid
    centroid = compute_centroid(reconstructed_points)

    # get x and y distributions shifted by centroid
    x_dist = [
        point[0] - centroid[0]
        for point in reconstructed_points
    ]

    y_dist = [
        point[1] - centroid[1]
        for point in reconstructed_points
    ]

    # get z distribution (z is compared relative to z=0 not the centroid)
    z_dist = [
        point[2]
        for point in reconstructed_points
    ]

    # define cutoff values
    x_cutoff = np.std(x_dist) * num_stdev
    y_cutoff = np.std(y_dist) * num_stdev
    z_cutoff = np.percentile(z_dist, z_percentile)

    # only retain point if it is within all cutoffs
    return [
        reconstructed_points[i]
        for i in range(len(reconstructed_points))
        if  np.abs(x_dist[i]) < x_cutoff
        and np.abs(y_dist[i]) < y_cutoff
        and z_dist[i] < z_cutoff
    ]

def compute_centroid(points):
    centroid = np.zeros(points[0].shape)
    for point in points:
        centroid += point / len(points)

    return centroid

def shift_points_to_centroid(points):
    centroid = compute_centroid(points)
    return [
        point - centroid
        for point in points
    ]

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

    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
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

    print(k_mats)

    images = [
        undistort(image, k, d, k_adj, roi)
        for (image, k_adj, roi) in zip(images, k_mats, rois)
    ]

    plt_utils.show_images(*images)
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
        for image in images
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
    
    images = [
        image
        for (image, is_valid) in zip(images, is_valids)
        if is_valid
    ]

    plt_utils.plot_image_points(images, image_points, titles=titles, sup_title="Image Chessboard Points")
    plt_utils.plot_point_path(images, corners, image_points, titles=titles, sup_title="Corner Point Sequence")

    #########################
    # get extrinsic camera params
    #########################
    
    length = 9
    width = 6
    square_size = 1.9
    G_mats = [
        get_camera_extrinsic_matrix_nls(points, k_mat, length=length, width=width, square_size=square_size)
        for (points, k_mat) in zip(image_points, k_mats)
    ]

    # print(G_mats)

    #########################
    # get point groups from chessboard corner image points
    #########################

    feature_groups = [
        [
            SIFTFeature(j, image_points[j][i], np.zeros(128))
            for j in range(len(image_points))
        ]
        for i in range(len(image_points[0]))
    ]

    # print(np.shape(feature_groups))

    proj_mats = [
        K @ G
        for (K, G) in zip(k_mats, G_mats)
    ]

    reconstructed_points = reconstruct_3D_points(feature_groups, proj_mats)

    #########################
    # error analysis
    #########################
    true_points = get_world_points(length, width, square_size)

    #convert from homogeneous coordinates
    true_points = [
        point[:3] / point[3]
        for point in true_points
    ]

    abs_error = [
        linalg.get_euclidean_distance(reco, point)
        for (reco, point) in zip(reconstructed_points, true_points)
    ]

    MAE = np.mean(abs_error)

    for (reco, point) in zip(reconstructed_points, true_points):
        print(reco, point)

    plt.figure()
    plt.hist(abs_error, label="Mean Absolute Error = %.2f" % MAE)
    plt.xlabel("Absolute Error")
    plt.ylabel("Number of Occurences")
    plt.title("Absolute Error of Reconstructed Chessboard Points")
    plt.legend()

    # # try again but remove error prone images
    good_indices = [0, 2, 3, 6, 7, 9, 10, 11, 13, 14]

    # all combinations of 2 images
    combinator = combinations(good_indices, 2)
    combination_list = list(combinator)

    errors = []
    for combination in combination_list:
        reduced_feature_groups = [
            [
                feature
                for feature in group
                if feature.image_idx in combination
            ]
            for group in feature_groups
        ]

        print(len(reduced_feature_groups[0]))

        reconstructed_points = reconstruct_3D_points(reduced_feature_groups, proj_mats)

        abs_error = [
            linalg.get_euclidean_distance(reco, point)
            for (reco, point) in zip(reconstructed_points, true_points)
        ]

        errors += abs_error

    MAE = np.mean(errors)
    
    plt.figure()
    plt.hist(errors, label="Mean Absolute Error = %.3f" % MAE, bins=30)
    plt.xlabel("Absolute Error (cm)")
    plt.ylabel("Number of Occurences")
    plt.title("Reconstruction Error for all Passable 2 Image Combinations")
    plt.legend()


    # feature_groups = [
    #     [
    #         feat
    #     ]
    # ]

    # k_mats = [
    #     k_mats[i]
    #     for i in good_indices
    # ]

    # G_mats = [
    #     G_mats[i]
    #     for i in good_indices
    # ]

    # # print(np.shape(feature_groups))

    # reconstructed_points = reconstruct_3D_points(feature_groups, proj_mats)

    # #########################
    # # error analysis
    # #########################
    # true_points = get_world_points(length, width, square_size)

    # #convert from homogeneous coordinates
    # true_points = [
    #     point[:3] / point[3]
    #     for point in true_points
    # ]

    # abs_error = [
    #     linalg.get_euclidean_distance(reco, point)
    #     for (reco, point) in zip(reconstructed_points, true_points)
    # ]

    # MAE = np.mean(abs_error)

    # for (reco, point) in zip(reconstructed_points, true_points):
    #     print(reco, point)

    # plt.figure()
    # plt.hist(abs_error, label="Mean Absolute Error = %.2f" % MAE)
    # plt.xlabel("Absolute Error")
    # plt.ylabel("Number of Occurences")
    # plt.title("Absolute Error of Reconstructed Chessboard Points")
    # plt.legend()

    plt.show()