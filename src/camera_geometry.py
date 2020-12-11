import numpy as np
import utils.linalg_utils as linalg
import cv2
from scipy.optimize import minimize

def get_world_points(length, width, square_size):
    '''
    @brief  Computes the 3D coordinates of the x-corners according to the 
            sorting order

    @param length       The number of x-corners along the longer side before 
                        the middle section is removed (for the image provided 
                        in images/chess_pattern.png, the length is 9)
    @param width        The number of x-corners along the shorter side before 
                        the middle section is removed (for the image provided 
                        in images/chess_pattern.png, the length is 6)
    @param square_size  The side length, in cm, of the chessboard square
    @return             A numpy array containing the world points in 
                        homogeneous coordinates
    '''
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

def reconstruct_extrinsic_matrix_from_parameters(extrinsic_params):
    '''
    @brief  Computes the extrinsic parameters matrix [R | t]

    @param extrinsic_params The set of extrinsic calibration params in a (6,) 
                            shape ndarray. The array should be formatted as
                            [alpha, beta, gamma, tx, ty, tz]
    @return                 the 3x4 extrinsic parameters matrix
    '''

    # construct transformation matrix
    alpha, beta, gamma = extrinsic_params[0:3]
    
    t = extrinsic_params[3:6].reshape(3,1)
    R = linalg.rotation_mat_3D(alpha, beta, gamma)
    
    return np.concatenate((R, t), axis=1)

def minimizer(func, starting_params, args=()):
    '''
    @brief Implements Powel minimization on the given function

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

    return results.x

def get_squared_error_sum(extrinsic_params, k, world_points, image_points):
    '''
    @brief  Computes the sum of squared projection errors.

    @param extrinsic_params The set of extrinsic calibration params in a (6,) 
                            shape ndarray. The array should be formatted as
                            [alpha, beta, gamma, tx, ty, tz]
    @param k                The 3x3 intrinsic calibration parameter matrix
    @param world_points     The coordinates of the x-corner 3D world points
    @param image_points     The actual 2D image points of detected x-corners
    @return                 The sum of squared projection errors as a float
    '''

    # extrinsic calibration matrix
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

def get_camera_extrinsic_matrix_nls(image_points, k, length=9, width=6, square_size=1.9):
    '''
    @brief  Uses a nonlinear least squares (NLS) method to compute the 
            extrinsic camera calibration matrix

    @param image_points The actual 2D image points of detected x-corners
    @param k            The 3x3 intrinsic calibration parameter matrix
    @param length       The number of x-corners along the longer side before 
                        the middle section is removed (for the image provided 
                        in images/chess_pattern.png, the length is 9)
    @param width        The number of x-corners along the shorter side before 
                        the middle section is removed (for the image provided 
                        in images/chess_pattern.png, the length is 6)
    @param square_size  The side length, in cm, of the chessboard square
    @return             The computed 3x4 extrinsic calibration matrix
    '''
    world_points = get_world_points(length, width, square_size)

    # use Powel's method to minimize squared error sum
    args = (k, world_points, image_points)
    initial_guess = np.array([0, 0, 0, 0, 0, 30])
    params = minimizer(get_squared_error_sum, initial_guess, args=args)

    return reconstruct_extrinsic_matrix_from_parameters(params)

def find_high_error_proj_mat_indices(image_points, P_mats, 
                                     mean_error_threshold=10, 
                                     max_error_threshold=20, length=9, 
                                     width=6, square_size=1.9):
    '''
    @brief  Uses a nonlinear least squares (NLS) method to compute the 
            extrinsic camera calibration matrix

    @param image_points         The actual 2D image points of detected 
                                x-corners
    @param P_mats               The 3x4 camera projection matrix 
                                (P= K[R | t])
    @param mean_error_threshold The maximum allowable mean projection error
    @param max_error_threshold  The maximum allowable max projection error
    @param length               The number of x-corners along the longer side 
                                before the middle section is removed (for the 
                                image provided in images/chess_pattern.png, 
                                the length is 9)
    @param width                The number of x-corners along the shorter side 
                                before the middle section is removed (for the 
                                image provided in images/chess_pattern.png, 
                                the length is 6)
    @param square_size          The side length, in cm, of the chessboard 
                                square
    @return                     The computed 3x4 extrinsic calibration matrix
    '''
    x_corners = get_world_points(length, width, square_size)

    # project x corners using projection matrix
    projections = [
        [
            P @ vec
            for vec in x_corners
        ]
        for P in P_mats 
    ]

    # convert from homogeneous coordinates
    projections = [
        [
            point[:2].reshape(2,) / point[2]
            for point in points
        ]
        for points in projections
    ]

    # compute mean and max projection errors for every projection matrix
    projection_errors = [
        [
            linalg.get_euclidean_distance(proj, point)
            for (proj, point) in zip(proj_points, points)
        ]
        for (proj_points, points) in zip(projections, image_points)
    ]

    mean_errors = [
        np.mean(error_dist)
        for error_dist in projection_errors
    ]

    max_errors = [
        np.max(error_dist)
        for error_dist in projection_errors
    ]

    # filter based on thresholds
    return [
        i
        for i in range(len(P_mats)-1, -1, -1)
        if mean_errors[i] > mean_error_threshold
        or max_errors[i] > max_error_threshold
    ]