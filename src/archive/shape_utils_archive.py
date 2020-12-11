from utils.common_types import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

def generate_octahedron():
    octahedron = [
        Face3D((1,0,0), (0,1,0), (0,0,1)),
        Face3D((1,0,0), (0,0,-1), (0,1,0)),
        Face3D((1,0,0), (0,0,1), (0,-1,0)),
        Face3D((1,0,0), (0,-1,0), (0,0,-1)),
        Face3D((-1,0,0), (0,0,1), (0,1,0)),
        Face3D((-1,0,0), (0,1,0), (0,0,-1)),
        Face3D((-1,0,0), (0,-1,0), (0,0,1)),
        Face3D((-1,0,0), (0,0,-1), (0,-1,0)),
    ]

    vertex_type = GL_TRIANGLES

    return (octahedron, vertex_type)

def generate_cube():
    cube = [
        Face3D((-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)),
        Face3D((-1,-1,-1), (-1,1,-1), (1,1,-1), (1,-1,-1)),
        Face3D((1,-1,-1), (1,1,-1), (1,1,1), (1,-1,1)),
        Face3D((-1,-1,-1), (-1,-1,1), (-1,1,1), (-1,1,-1)),
        Face3D((-1,1,-1), (-1,1,1), (1,1,1), (1,1,-1)),
        Face3D((-1,-1,-1), (1,-1,-1), (1,-1,1), (-1,-1,1)),
    ]

    vertex_type = GL_QUADS

    return (cube, vertex_type)

def generate_sphere(num_circles=3, dphi=np.pi/4):
    top_hemisphere = [np.array([0, 1, 0])]
    
    circle_y_vals = [
        1/(num_circles+1) * i
        for i in range(num_circles)
    ]

    circle_phi_vals = np.linspace(0, 2*np.pi, int(round(2*np.pi/dphi)))

    for y in circle_y_vals:
        r = np.sqrt(1-y**2)
        for phi in circle_phi_vals:
            top_hemisphere.append(np.array([r*np.cos(phi), y, r*np.sin(phi)]))
    
    vertex_type = GL_POLYGON

    return (top_hemisphere, vertex_type)