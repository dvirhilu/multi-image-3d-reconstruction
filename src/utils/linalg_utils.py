import numpy as np
from scipy.linalg import norm, inv

##################################
# Coordinate System Conversions
##################################

def cyl_2_cart(r, phi, z):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y, z)

def cart_2_cyl(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi, z)

def pol_2_cart2d(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y)

def cart2d_2_pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi)

##################################
# Vector Operations
##################################

def unit_vect(vector):
    return vector/norm(vector)

def component_in_dir(vector, direction):
    return np.dot(vector, direction)/norm(direction)

def proj_2_vec(orig_vector, proj_direction):
    return component_in_dir(orig_vector, proj_direction) * unit_vect(proj_direction)

def proj_2_plane(vector, plane_normal):
    return vector - proj_2_vec(vector, plane_normal)

def proj_2_xy(vector):
    proj = proj_2_plane(vector, np.array([0, 0, 1]))
    return np.array([proj[0], proj[1]])

def rotate_vec(vec, theta, rot_axis):
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
    return norm(vec2 - vec1)
    
def get_vec_cos_angle(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def cross_2d(vec1, vec2):
    return vec1[0]*vec2[1] - vec1[1]*vec2[0]

##################################
# Matrix Operations
##################################(

def pseudo_inv(A):
    gramian = A.T @ A
    inv_gramian = inv(gramian)

    return inv_gramian @ A.T

def rotation_mat_3D(alpha, beta, gamma):
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