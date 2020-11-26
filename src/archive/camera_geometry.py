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

def get_initial_params(world_points, image_points, camera_mat):
    # camera position and orientation vector can be rearranged as A@camera_extrinsics_matrix.reshape(1,12)
    # here, columns related to Z are ommitted since they are linearly dependent (chessboard points lie on a plane)
    # need to later use the rotation coeffcients R[0:2, :] to retrieve R[3,:]
    A = get_worldpoint_lin_sys_matrix(world_points, camera_mat)

    # generate image_points vector
    image_points_vector = get_image_points_vector(image_points)

    # calculate partial (missing 3rd column due to Z omission) extrinsic camera matrix using least squares solution
    pseudo_inverse = linalg.pseudo_inv(A)
    partial_mat = (pseudo_inverse @ image_points_vector).reshape(3,3)

    return extract_params_from_partial_extrinsic_mat(partial_mat)
