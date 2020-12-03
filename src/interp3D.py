import numpy as np
from scipy.spatial import Delaunay
from utils.common_types import Face3D

if __name__=="__main__":
    num_phi = 10
    num_theta = 10
    phi_vals = np.linspace(0, 2*np.pi, num_phi)
    theta_vals = np.linspace(0, np.pi, num_theta)
    r = 1

    sphere = [] 
    for theta in theta_vals:
        for phi in phi_vals:
            x = r*np.cos(theta)*np.cos(phi)
            y = r*np.cos(theta)*np.sin(phi)
            z = r*np.sin(theta)
            sphere.append([x, y, z])

    sphere = np.array(sphere)

    delaunay = Delaunay(sphere[:, 0:2])

    print(np.shape(delaunay.simplices))

    sphere = []
    for simplex in delaunay.simplices:
        