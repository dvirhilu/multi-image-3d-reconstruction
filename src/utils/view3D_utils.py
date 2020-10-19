'''
Multi-Image 3D Reconstruction
3D Viewer Utils

This script contains a set of useful helper functions to 
allow the reconstructed object to be viewed in 3D rotated
view.

Author:   Dvir Hilu
Date:     14/10/2020

sources:
Math for Programmers: 3D graphics, machine learning, and simulations with Python MEAP V10 by Paul Orland
'''

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
import numpy as np
import math_utils
from common_types import *

blue_cmap = cm.get_cmap("Blues")

def get_shade_val(face, light_dir):
    unit_normal = math_utils.unit_vect(face.normal_vector)
    unit_light_dir = math_utils.unit_vect(light_dir)
    shade_inv = np.dot(unit_normal, unit_light_dir)

    return 1-shade_inv

def set_face_render_info(face_list, light_dir, cmap, cline = None):
    for face in face_list:
        cfill = cmap(get_shade_val(face, light_dir))
        face.set_render_colours(cfill, cline)

def render_face(face, ax):
    polygon = Polygon(face.vertices_xy, True)
    patches = PatchCollection([polygon], facecolors=face.cfill, edgecolors=face.cline)

    ax.add_collection(patches)

def render_faces(ax, face_list, cmap, light_dir= np.array([1,1,1]), cline=None):
    # set information about fill and line colours for faces
    set_face_render_info(face_list, light_dir, cmap, cline=cline)

    face_list.sort(key= lambda face:face.normal_vector[2])
    for face in face_list:
        render_face(face, ax)

def set_range(ax, face_list, padding_percentage = 5, aspect_ratio=1):
    x_vals = [vertex[0] for face in face_list for vertex in face.vertices_xy]
    y_vals = [vertex[1] for face in face_list for vertex in face.vertices_xy]

    xmin = min(x_vals)
    xmax = max(x_vals)
    ymin = min(y_vals)
    ymax = max(y_vals)

    padding_factor = 1 + padding_percentage/100

    ax.set_xlim([padding_factor*xmin, padding_factor*xmax])
    ax.set_ylim([padding_factor*ymin, padding_factor*ymax])

    ratio_default=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0])
    ax.set_aspect(ratio_default*aspect_ratio)

def draw_octahedron():
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

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7,7))

    set_range(ax1, octahedron)
    render_faces(ax1, octahedron, cmap = blue_cmap, cline='k')

    for face in octahedron:
        face.rotate(angle=np.pi/6, rot_axis=np.array([0,1,0]))

    set_range(ax2, octahedron)
    render_faces(ax2, octahedron, cmap = blue_cmap, cline='k')

    plt.show()

def draw_cube():
    cube = [
        Face3D((-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)),
        Face3D((-1,-1,1), (-1,1,1), (1,1,1), (1,-1,1)),
        Face3D((1,-1,-1), (1,1,-1), (1,1,1), (1,-1,1)),
        Face3D((-1,-1,-1), (-1,-1,1), (-1,1,1), (-1,1,-1)),
        Face3D((-1,1,-1), (-1,1,1), (1,1,1), (1,1,-1)),
        Face3D((-1,-1,-1), (1,-1,-1), (1,-1,1), (-1,-1,1)),
    ]

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7,7))

    for face in cube:
        face.rotate(angle=np.pi/6, rot_axis=np.array([1,-1,1]))

    set_range(ax1, cube)
    render_faces(ax1, cube, cmap = blue_cmap, cline='k')

    for face in cube:
        face.rotate(angle=np.pi/6, rot_axis=np.array([0,1,0]))

    set_range(ax2, cube)
    render_faces(ax2, cube, cmap = blue_cmap, cline='k')

    plt.show()