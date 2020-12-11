import numpy as np
from scipy.linalg import norm, inv, eig

##################################
# Coordinate System Conversions
##################################

def cart2d_2_pol(x, y):
    '''
    @brief  Converts the x,y coordinate into polar coordinates
    
    @param x    x cartesian coordinate
    @param y    y cartesian coordinate

    @returns r      The r polar coordinate
    @returns phi    The phi polar coordinate
    '''
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi)

##################################
# Vector Operations
##################################

def unit_vect(vector):
    '''
    @brief  computes the unit vector in the direction of the inputted vector
    
    @param vector   direction of the desired unit vector

    @returns        A vector of magnitude 1 in inputted direction
    '''
    return vector/norm(vector)

def rotate_vec(vec, theta, rot_axis):
    '''
    @brief  Rotates a vector in 3D
    
    @param vec      The vector to be rotated
    @param theta    The rotation angle
    @param rot_axis A vector pointing in the direction of the rotation axis

    @returns        the rotated vecror
    '''
    axis = unit_vect(rot_axis)
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    r_11 = np.cos(theta) + ux**2*(1-np.cos(theta))
    r_12 = ux*uy*(1-np.cos(theta)) - uz*np.sin(theta)
    r_13 = ux*uz*(1-np.cos(theta)) + uy*np.sin(theta)
    r_21 = uy*ux*(1-np.cos(theta)) + uz*np.sin(theta)
    r_22 = np.cos(theta) + uy**2*(1-np.cos(theta))
    r_23 = uy*uz*(1-np.cos(theta)) - ux*np.sin(theta)
    r_31 = uz*ux*(1-np.cos(theta)) - uy*np.sin(theta)
    r_32 = uz*uy*(1-np.cos(theta)) + ux*np.sin(theta)
    r_33 = np.cos(theta) + uz**2*(1-np.cos(theta))

    R = np.array([
            [r_11, r_12, r_13],
            [r_21, r_22, r_23],
            [r_31, r_32, r_33]
        ])

    return R @ vec

def get_euclidean_distance(vec1, vec2):
    '''
    @brief  computes the euclidean distance between two vectors
    
    @param vec1 The first vector in the computation
    @param vec2 The second vector in the computation

    @returns    the euclidean distance between vec1 and vec2
    '''
    return norm(vec2 - vec1)
    
def get_vec_cos_angle(vec1, vec2):
    '''
    @brief  computes the angle between two vectors
    
    @param vec1 The first vector in the computation
    @param vec2 The second vector in the computation

    @returns    the angle between vec1 and vec2
    '''
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def cross_2d(vec1, vec2):
    '''
    @brief  computes "2D cross product". That is, it computes the determinant 
            of the matrix [vec1 | vec2]
    
    @param vec1 The first vector in the computation
    @param vec2 The second vector in the computation

    @returns    the value of the determinant of [vec1 | vec2]
    '''
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]

##################################
# Matrix Operations
##################################

def gramian(A):
    '''
    @brief  Computes the Gramian of a matrix, defined as (A^T A)

    @param A    The matrix used in the computation
    @returns    The Gramian of matrix A
    '''
    return A.T @ A

def pseudo_inv(A):
    '''
    @brief  Computes the pseudo-inverse of a matrix, defined as 
            (A^T A)^(-1)A^T

    @param A    The matrix used in the computation
    @returns    The pseudo-inverse of matrix A
    '''
    gramian_mat = gramian(A)
    inv_gramian = inv(gramian_mat)

    return inv_gramian @ A.T

def rotation_mat_3D(alpha, beta, gamma):
    '''
    @brief  Computes a 3D rotation matrix

    @param alpha    The alpha Euler angle
    @param beta     The beta Euler angle
    @param gamma    The gamma Euler angle
    @returns        A matrix that rotates a vector by the given Euler angles
    '''
    r11 = np.cos(alpha)*np.cos(beta)
    r12 = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
    r13 = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
    r21 = np.sin(alpha)*np.cos(beta)
    r22 = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
    r23 = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
    r31 = -np.sin(beta)
    r32 = np.cos(beta)*np.sin(gamma)
    r33 = np.cos(beta)*np.cos(gamma)

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

def solve_homogeneous_least_squares(A):
    '''
    @brief  Computes the solution to a homogeneous least squares problem with 
            least squares matrix A

    @param A    The least squares matrix of the system
    @returns    The least squares solution of the homogeneous system
    '''
    gramian_mat = gramian(A)
    
    (w, v) = eig(gramian_mat)

    i_min = np.argmin(w)

    return v[:, i_min]

##################################
# Centroid Calculations
##################################

def compute_intensity_centroid(window):
    '''
    @brief  Computes the centroid of intensity values in an image window

    @param window           The image window used for the computation
    @returns y_centroid     The y coordinate of the centroid
    @returns x_centroid     The x coordinate of the centroid
    '''

    rows, cols = window.shape[:2]

    total_mass = np.sum(window)
    y_centroid = 0
    x_centroid = 0
    for i in range(rows):
        for j in range(cols):
            mass = window[i,j]
            y_centroid += i*mass / total_mass
            x_centroid += j*mass / total_mass
    
    return(y_centroid, x_centroid)

def compute_point_group_centroid(points):
    '''
    @brief  Computes the centroid of a point cloud

    @param points   The point cloud used for the computation
    @returns        A point with the same shape as the points in the cloud 
                    corresponding to the centroid        
    '''
    centroid = np.zeros(points[0].shape)
    for point in points:
        centroid += point / len(points)

    return centroid