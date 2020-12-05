import numpy as np
import utils.file_io_utils as io
import utils.linalg_utils as linalg
import utils.plt_utils as plt_utils
from process_image_background import get_ordered_image_points, get_undistored_k_matrix, undistort
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_world_points(length, width, square_size):
    xmax = square_size*(length-1)
    ymax = square_size*(width-1)
    a = square_size
    # in each cluster, extreme corner comes first with clockwise ordering
    return [
        # top left corner cluster
        np.array([0,        ymax,       0,      1]).reshape(4,1),
        np.array([a,        ymax,       0,      1]).reshape(4,1),
        np.array([a,        ymax-a,     0,      1]).reshape(4,1),
        np.array([0,        ymax-a,     0,      1]).reshape(4,1),
        # top right corner cluster
        np.array([xmax,     ymax,       0,      1]).reshape(4,1),
        np.array([xmax,     ymax-a,     0,      1]).reshape(4,1),
        np.array([xmax-a,   ymax-a,     0,      1]).reshape(4,1),
        np.array([xmax-a,   ymax,       0,      1]).reshape(4,1),
        # bottom right corner cluster
        np.array([xmax,     0,          0,      1]).reshape(4,1),
        np.array([xmax-a,   0,          0,      1]).reshape(4,1),
        np.array([xmax-a,   a,          0,      1]).reshape(4,1),
        np.array([xmax,     a,          0,      1]).reshape(4,1),
        # bottom left corner cluster
        np.array([0,        0,          0,      1]).reshape(4,1),
        np.array([0,        a,          0,      1]).reshape(4,1),
        np.array([a,        a,          0,      1]).reshape(4,1),
        np.array([a,        0,          0,      1]).reshape(4,1)
    ]

def get_single_worldpoint_matrix(world_point, camera_mat):
    x, y = world_point[:2, 0]
    
    # remove z cols since they become linearly dependent. solve for later
    A = np.array([
        [x, y, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, x, y, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, x, y, 1]
    ])

    return camera_mat @ A

def extract_params_from_partial_extrinsic_mat(partial_mat):
    partial_rot_mat = partial_mat[:, 0:2]
    t1, t2, t3 = partial_mat[:, 2]

    # matrix is scaled. Find scaling factor by checking square sum of column 1 (original should be 1)
    col_square_sum = np.sum(partial_rot_mat[:, 0]**2)
    scale_factor = np.sqrt(col_square_sum)

    normalized_partial_rot_mat = partial_rot_mat / scale_factor

    # R31 = -sin(beta)
    beta = np.arcsin(-normalized_partial_rot_mat[2, 0])

    # R21 = sin(alpha)cos(beta)
    alpha = np.arcsin(normalized_partial_rot_mat[1, 0] / np.cos(beta))

    # R32 = cos(beta)sin(gamma)
    gamma = np.arcsin(normalized_partial_rot_mat[2, 1] / np.cos(beta))

    return np.array([alpha, beta, gamma, t1, t2, t3])

def reconstruct_extrinsic_matrix_from_parameters(extrinsic_params):
    # construct transformation matrix
    alpha, beta, gamma = extrinsic_params[0:3]
    
    t = extrinsic_params[3:6].reshape(3,1)
    R = linalg.rotation_mat_3D(alpha, beta, gamma)
    
    return np.concatenate((R, t), axis=1)

def minimizer(func, starting_params, args=()):
    '''
    @brief Implements nelder mead optimization on the given function

    For more details on the implementation, visit
    https://github.com/ubc-subbots/sound-localization-simulator/blob/master/docs/Position_Calculation_Algorithms.pdf

    @param func             The function needing to be minimized
    @param starting_params  The initial guess for the function parameters that will result
                            in the function being minimized. The function expects this to
                            be inputted as a numpy array
    @param args             A tuple containing any other arguments that should be inputted
                            to the function (constants, non-numerical parameters, etc.)
    @return                 A numpy array containing the value of arguments that will minimize 
                            func
    '''
    results = minimize(func, starting_params, args=args, method="Powell")

    if (not results.success):
        print(results.message)

    print("initial guess: ", starting_params)
    print("sum of squares: ", func(starting_params, *args))
    print("Converged with %d iterations" %results.nit)
    print("results: ", results.x)
    print("sum of squares: ", func(results.x, *args))

    return results.x

def get_squared_error_sum(extrinsic_params, k, world_points, image_points):
    G = reconstruct_extrinsic_matrix_from_parameters(extrinsic_params)

    # projection matrix
    P = (k @ G)

    expected_image_points = [
        P @ world_point
        for world_point in world_points
    ]

    # convert from homogeneous coordinates
    expected_image_points = [
        point[:2,0] / point[2]
        for point in expected_image_points
    ]

    squared_error = [
        linalg.get_euclidean_distance(expected_image_point, image_point)**2
        for (expected_image_point, image_point) in zip(expected_image_points, image_points)
    ]

    return sum(squared_error)

def get_camera_extrinsic_matrix_nls(image_points, camera_mat, length=9, width=6, square_size=1.9):
    world_points = get_world_points(length, width, square_size)

    # use nelder mead to minimize squared error sum
    args = (camera_mat, world_points, image_points)
    # initial_guess = get_initial_params(world_points, image_points, camera_mat)
    initial_guess = np.array([0, 0, 0, 0, 0, 30])

    params = minimizer(get_squared_error_sum, initial_guess, args=args)

    return reconstruct_extrinsic_matrix_from_parameters(params)

def compute_fundemental_matrix(K1, K2, G1, G2):
    # compute camera matrices
    P1 = K1 @ G1
    P2 = K2 @ G2

    # compute second epipole
    R1 = G1[:, 0:3]
    T1 = G1[:, 3].reshape(3,1)
    C1 = -R1.T @ T1
    C1 = np.append(C1, 1).reshape(4,1)
    e2 = P2 @ C1

    print(C1)
    print(e2)

    # compute fundemental matrix components
    P1_inv = linalg.pseudo_inv(P1)
    e2_skew = linalg.skew_sym(e2.flatten())

    return e2_skew @ P2 @ P1_inv

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

    print(G_mats)

    #########################
    # compare re-projection
    #########################
    chessboard_points = get_world_points(length, width, square_size)

    projections = [
        [
            k_mat @ G @ vec
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

    plt_utils.plot_image_points(images[5:7], projections[5:7], titles=titles[5:7], sup_title="True Chessboard Points Projected Onto Images", same_colour=False)

    #########################
    # error analysis
    #########################
    MAE = [
        np.mean([
            linalg.get_euclidean_distance(proj, point)
            for (proj, point) in zip(proj_points, img_points)
        ])
        for (proj_points, img_points) in zip(projections, image_points)
    ]

    max_err = [
        np.max([
            linalg.get_euclidean_distance(proj, point)
            for (proj, point) in zip(proj_points, img_points)
        ])
        for (proj_points, img_points) in zip(projections, image_points)
    ]

    for (proj, point) in zip(projections[13], image_points[13]):
        print(proj, point)

    plt.figure()
    plt.bar(range(len(images)), MAE)
    plt.xlabel("Image Index")
    plt.ylabel("Mean Absolute Error (pixels)")
    plt.title("Projection Mean Absolute Error")

    plt.figure()
    plt.bar(range(len(images)), max_err)
    plt.xlabel("Image Index")
    plt.ylabel("Maximum Absolute Error (pixels)")
    plt.title("Projection Maximum Absolute Error")

    # ###################### test fundemental matrix ######################################
    # images = images[:2]
    # K1 = k_mats[0]
    # K2 = k_mats[1]
    # G1 = G_mats[0]
    # G2 = G_mats[1]

    # F = compute_fundemental_matrix(K1, K2, G1, G2)

    # image_points = [
    #     [
    #         point
    #         for point in points[:2]
    #     ]
    #     for points in image_points[:2]
    # ]

    # point_vecs = [
    #     np.append(point, 1).reshape(3,1)
    #     for point in image_points[1]
    # ]

    # projective_lines = [
    #     F @ point_vec
    #     for point_vec in point_vecs
    # ]

    # plt_utils.plot_image_points(images, image_points)

    # x  = np.linspace(0, images[0].shape[1], 20)

    # l1 = projective_lines[0]
    # l2 = projective_lines[1]

    # print(l1)
    # print(l2)

    # y1 = -x*l1[0]/l1[1] - l1[2]/l1[1] 
    # y2 = -x*l2[0]/l2[1] - l2[2]/l2[1]

    # plt.plot(x,y1) 
    # plt.plot(x,y2) 

    plt.show()
