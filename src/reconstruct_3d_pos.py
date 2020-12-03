import numpy as np
import utils.linalg_utils as linalg
import utils.file_io_utils as io
from process_image_background import get_ordered_image_points, undistort, get_undistored_k_matrix
from utils import plt_utils
import cv2
from sift import SIFTFeature
from camera_geometry import get_camera_extrinsic_matrix_nls
import matplotlib.pyplot as plt

def create_ls_matrix(k_mats, G_mats, image_points):
    proj_matrices = tuple(
        k @ G
        for (k, G) in zip(k_mats, G_mats)
    )

    x_cross_prod_eqns = np.array([
        point[0]*P[2, :].T - P[0, :].T
        for (point, P) in zip(image_points, proj_matrices)
    ])

    y_cross_prod_eqns = np.array([
        point[1]*P[2, :].T - P[1, :].T
        for (point, P) in zip(image_points, proj_matrices)
    ])

    return np.concatenate((x_cross_prod_eqns, y_cross_prod_eqns), axis=0)

def reconstruct_3D_points(feature_groups, K_mats, G_mats):
    reconsturcted_points = []
    for group in feature_groups:
        relevant_K_mats = [
            K_mats[feature.image_idx]
            for feature in group
        ]
        relevant_G_mats = [
            G_mats[feature.image_idx]
            for feature in group
        ]

        image_points = [
            feature.pixel_location
            for feature in group
        ]

        ls_mat = create_ls_matrix(relevant_K_mats, relevant_G_mats, image_points)

        world_point = linalg.solve_homogeneous_least_squares(ls_mat)

        # convert back from homogeneous coordinates
        world_point = world_point[0:3].reshape(3,1) / world_point[3]

        reconsturcted_points.append(world_point)

    return reconsturcted_points

if __name__=="__main__":
    camera_calib = "SamsungGalaxyA8"
    
    # first, grab camera and distortion matrices
    k, d = io.load_calib_coefficients(camera_calib)

    # generate image points
    images = io.load_object_images("monkey_thing")
    # good_indices = [0, 2, 4, 5, 7, 10, 11]
    good_indices = [0, 2, 4, 7]
    # good_indices = [0, 2]
    images = [
        images[i] 
        for i in good_indices
    ]
    titles = [
        "Image %0d" %i
        for i in good_indices
    ]

    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]
    # plt_utils.show_images(*images)

    #########################
    # undistort images
    #########################
    undistort_tuples = [
        get_undistored_k_matrix(image, k, d)
        for image in images
    ]

    k_mats, rois = zip(*undistort_tuples)

    # print(k_mats)

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
    r=15
    p=0.4
    ret_tuples = [
        get_ordered_image_points(image, windowsize=windowsize, sobel_size=sobel_size, k=harris_const, harris_threshold=harris_threshold, r=r, p=p)
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

    plt_utils.plot_image_points(images, image_points)

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

    reconstructed_points = reconstruct_3D_points(feature_groups, k_mats, G_mats)

    print(reconstructed_points)

    chessboard_points = [
        np.array([
            i*square_size,
            j*square_size,
            0,
        ]).reshape(3,1)
        for i in range(length)
        for j in range(width)
    ]

    print(chessboard_points)

    # Compare reprojections
    projections = [
        [
            k_mat @ G @ np.array([vec[0,0], vec[1,0], vec[2,0], 1]).reshape(4,1)
            for vec in reconstructed_points
        ]
        for (k_mat, G) in zip(k_mats, G_mats) 
    ]

    # re-normalize
    projections = [
        [
            point[:2,0] / point[2]
            for point in points
        ]
        for points in projections
    ]

    plt_utils.plot_image_points(images, projections, titles=titles, sup_title="Reconstructed 3D Points Reprojections")

    # Compare reprojections
    projections = [
        [
            k_mat @ G @ np.array([vec[0,0], vec[1,0], vec[2,0], 1]).reshape(4,1)
            for vec in chessboard_points
        ]
        for (k_mat, G) in zip(k_mats, G_mats) 
    ]

    # re-normalize
    projections = [
        [
            point[:2,0] / point[2]
            for point in points
        ]
        for points in projections
    ]

    plt_utils.plot_image_points(images, projections)


    plt.show()