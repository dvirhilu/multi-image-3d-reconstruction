import numpy as np
import utils.file_io_utils as io
import utils.linalg_utils as linalg
import utils.plt_utils as plt_utils
from process_image_background import get_ordered_image_points
import cv2
import matplotlib.pyplot as plt

def get_world_points(length, width, square_size):
    xmax = length*square_size/2
    ymax = width*square_size/2
    # in each cluster, extreme corner comes first with clockwise ordering
    return [
        # top left corner cluster
        np.array([-xmax,                ymax,               1]),
        np.array([-xmax+square_size,    ymax,               1]),
        np.array([-xmax+square_size,    ymax-square_size,   1]),
        np.array([-xmax,                ymax-square_size,   1]),
        # top right corner cluster
        np.array([xmax,                 ymax,               1]),
        np.array([xmax,                 ymax-square_size,   1]),
        np.array([xmax-square_size,     ymax-square_size,   1]),
        np.array([xmax-square_size,     ymax,               1]),
        # bottom right corner cluster
        np.array([xmax,                 -ymax,              1]),
        np.array([xmax-square_size,     -ymax,              1]),
        np.array([xmax-square_size,     -ymax+square_size,  1]),
        np.array([xmax,                 -ymax+square_size,  1]),
        # bottom left corner cluster
        np.array([-xmax,                -ymax,              1]),
        np.array([-xmax,                -ymax+square_size,  1]),
        np.array([-xmax+square_size,    -ymax+square_size,  1]),
        np.array([-xmax+square_size,    -ymax,              1])
    ]

def get_single_worldpoint_matrix(world_point, camera_mat):
    x, y, z = tuple(world_point)
    
    # remove z cols since they become linearly dependent. solve for later
    A = np.array([
        [x, y, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, x, y, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, x, y, 1]
    ])

    return camera_mat @ A

def get_worldpoint_lin_sys_matrix(world_points, camera_mat):
    '''
    for a given worldpoint, matrix is
    |x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0| -> row1
    |0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0| -> row2
    |0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1| -> row3
    '''

    # for each world point, generate the matrix
    matrices = tuple(
        get_single_worldpoint_matrix(point, camera_mat)
        for point in world_points
    )

    return np.concatenate(matrices, axis=0)

def get_image_points_vector(image_points):
    '''
    to match format of worldpoint_lin_sys_matrix,
    column vector with len(image_points)*3 elements
    each 3 elements correspond to a single point in homogeneous coordinates: [x, y, 1]
    '''

    point_vecs = tuple(
        np.array([point[0], point[1], 1]).reshape(3, 1)
        for point in image_points
    )

    return np.concatenate(point_vecs, axis=0)

def reconstruct_geo_mat_from_partial_mat(partial_mat):
    partial_rot_mat = partial_mat[:, 0:2]
    trans_mat = partial_mat[:, 2].reshape(3, 1)

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

    # construct column 3
    col3 = np.array([
        [np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
        [np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
        [np.cos(beta)*np.cos(gamma)],
    ])

    # Denormalize column 3
    col3 *= scale_factor

    print(scale_factor, alpha, beta, gamma)

    return np.concatenate((partial_rot_mat, col3, trans_mat), axis=1)

def get_camera_extrinsic_matrix(image_points, camera_mat, length=9, width=6, square_size=1.9):
    world_points = get_world_points(length, width, square_size)

    # camera position and orientation vector can be rearranged as A@camera_extrinsics_matrix.reshape(1,12)
    # here, columns related to Z are ommitted since they are linearly dependent (chessboard points lie on a plane)
    # need to later use the rotation coeffcients R[0:2, :] to retrieve R[3,:]
    A = get_worldpoint_lin_sys_matrix(world_points, camera_mat)

    # generate image_points vector
    image_points_vector = get_image_points_vector(image_points)

    # calculate partial (missing 3rd column due to Z omission) extrinsic camera matrix using least squares solution
    pseudo_inverse = linalg.pseudo_inv(A)
    partial_mat = (pseudo_inverse @ image_points_vector).reshape(3,3)

    # construct 3rd column from known elements
    return reconstruct_geo_mat_from_partial_mat(partial_mat)


if __name__=="__main__":
    camera_calib = "SamsungGalaxyA8"
    length = 9
    width = 6
    square_size = 1.9
    
    # first, grab camera matrix
    k, d = io.load_calib_coefficients(camera_calib)

    print(k)

    world_points = get_world_points(length, width, square_size)

    A = get_worldpoint_lin_sys_matrix(world_points, k)

    # generate image points
    images = io.load_object_images("monkey_thing")
    # good_indices = [0, 2, 4, 5, 7, 10, 11]
    good_indices = [0, 2]
    images = [
        images[i] 
        for i in good_indices
    ]
    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]
    plt_utils.show_images(*images)

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

    is_valids = [
        ret_tuple[0]
        for ret_tuple in ret_tuples
    ]

    image_points = [
        ret_tuple[1]
        for (ret_tuple, is_valid) in zip(ret_tuples, is_valids)
        if is_valid
    ]

    corners = [
        ret_tuple[2]
        for (ret_tuple, is_valid) in zip(ret_tuples, is_valids)
        if is_valid
    ]
    
    images = [
        image
        for (image, is_valid) in zip(images, is_valids)
        if is_valid
    ]

    image_points_vectors = [
        get_image_points_vector(points)
        for points in image_points
    ]

    plt_utils.plot_image_points(images, image_points)

    G_mats = [
        get_camera_extrinsic_matrix(points, k)
        for points in image_points
    ]

    print(G_mats)

    # compare re-projection
    length = 9
    width = 6
    square_size = 1.9
    chessboard_points = [
        np.array([
            i*square_size - length*square_size/2,
            j*square_size - width*square_size/2,
            0,
            1
        ]).reshape(4,1)
        for i in range(length)
        for j in range(width)
    ]

    projections = [
        [
            k @ G @ vec
            for vec in chessboard_points
        ]
        for G in G_mats 
    ]

    plt_utils.plot_image_points(images, projections)

    plt.show()
