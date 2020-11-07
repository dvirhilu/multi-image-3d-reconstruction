import numpy as np
from numpy.linalg import norm

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

    return np.matmul(R, vec)
    

    
