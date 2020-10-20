import numpy as np
from enum import Enum
import math_utils

class CoordinateRepr(Enum):
    CARTESIAN = 0,
    CYLINDRICAL = 1,
    SPHERICAL = 2

class Face3D:

    def __init__(self, *vertices):
        self.update(*vertices)

    def print_face3D(self):
        v1 = self.vertices[0]
        v2 = self.vertices[1]
        v3 = self.vertices[2]
        string =  ("3D Face(\n" + 
                    "[" + str(v1[0]) + ", " + str(v1[1]) + ", " + str(v1[2]) + "]\n" + 
                    "[" + str(v2[0]) + ", " + str(v2[1]) + ", " + str(v2[2]) + "]\n" + 
                    "[" + str(v3[0]) + ", " + str(v3[1]) + ", " + str(v3[2]) + "]\n" + 
                    ")\n")
        print(string)

    def print_face2D(self):
        v1 = self.vertices_xy[0]
        v2 = self.vertices_xy[1]
        v3 = self.vertices_xy[2]
        string =  ("3D Face(\n" + 
                    "[" + str(v1[0]) + ", " + str(v1[1]) + "]\n" + 
                    "[" + str(v2[0]) + ", " + str(v2[1]) + "]\n" + 
                    "[" + str(v3[0]) + ", " + str(v3[1]) + "]\n" + 
                    ")\n")
        print(string)


    def update(self, *vertices):
        self.vertices = [np.array(vertex) for vertex in vertices]

        vec1 = self.vertices[1] - self.vertices[0]
        vec2 = self.vertices[2] - self.vertices[0]
        self.normal_vector = np.cross(vec1, vec2)

        for vertex in self.vertices[1:]:
            if np.abs(np.dot(vertex-self.vertices[0], self.normal_vector)) > 1e-3:
                print(vertex-self.vertices[0], self.normal_vector, np.dot(vertex-self.vertices[0], self.normal_vector))
                raise ValueError("Verteces are not co-planar")

        self.vertices_xy = [math_utils.proj_2_xy(vertex) for vertex in self.vertices]


    def to_2D(self, plane_normal):
        return [math_utils.proj_2_plane(vertex, plane_normal) for vertex in self.vertices]

    def set_render_colours(self, cfill):
        self.cfill = cfill

    def rotate(self, angle, rot_axis):
        vertices = [math_utils.rotate_vec(vertex, angle, rot_axis)
                    for vertex in self.vertices]
        self.update(*vertices)