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

def get_shade_val(face, light_dir=np.array([1,1,1])):
    unit_normal = math_utils.unit_vect(face.normal_vector)
    unit_light_dir = math_utils.unit_vect(light_dir)
    shade_inv = np.dot(unit_normal, unit_light_dir)

    return 1-shade_inv

def set_face_render_info(face_list, cmap=blue_cmap, light_dir=np.array([1,1,1])):
    for face in face_list:
        cfill = cmap(get_shade_val(face, light_dir=light_dir))
        face.set_render_colours(cfill)

def render_face(face):
    # polygon = Polygon(face.vertices_xy, True)
    # patches = PatchCollection([polygon], facecolors=face.cfill, edgecolors=face.cline)

    # ax.add_collection(patches)
    color = face.cfill
    for vertex in face.vertices:
        glColor3fv((color[0], color[1], color[2]))
        glVertex3fv(vertex)

def render_faces(face_list, cmap=blue_cmap, light_dir= np.array([1,1,1])):
    # set information about fill and line colours for faces
    set_face_render_info(face_list, cmap=cmap, light_dir=light_dir)

    # face_list.sort(key= lambda face:face.normal_vector[2])
    for face in face_list:
        render_face(face)

def init_window(windowsize=(400,400)):
    pygame.init()
    window = pygame.display.set_mode(windowsize, DOUBLEBUF|OPENGL)
    return window

def set_opengl_params(perspective_distance=5):
    gluPerspective(45, 1, 0.1, 50.0)
    glTranslatef(0.0,0.0, -perspective_distance)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glCullFace(GL_BACK) 

def update_screen(face_list, cmap=blue_cmap, gl_mode=GL_POINTS):
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glBegin(gl_mode)
    render_faces(face_list, cmap=cmap)
    glEnd()
    pygame.display.flip()

def rotate_view(face_list, angle, x, y, z, use_opengl=False):
    if use_opengl:
        glRotatef(angle, x, y, z)
    else:
        for face in face_list:
            angle_rad = np.pi/180 * angle
            face.rotate(angle=angle_rad, rot_axis=np.array([x,y,z]))

def is_mouse_down(event, mouse_down):
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        return True
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        return False
    else:
        return mouse_down

def should_obj_rotate(event, mouse_down):
    return event.type == pygame.MOUSEMOTION and mouse_down

def execute_mouse_controlled_rotation(face_list, x, y, x_prev, y_prev):
    # mouse coordinates increase y in downward direction
    displacement_vec = np.array([x-x_prev, -(y-y_prev), 0])
    if math_utils.norm(displacement_vec) == 0:
        return
    rotation_axis = np.cross(displacement_vec, np.array([0,0,-1]))
    rotation_angle = 0.3*math_utils.norm(displacement_vec)
    
    rotate_view(face_list, rotation_angle, *rotation_axis)
        

def draw_octahedron(windowsize=(700,700), gl_mode=GL_TRIANGLES, rotate=False, 
                    use_opengl=False, rotations_per_s = 0.1, rotation_axis=(0,0,1)):
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

    degrees_per_s = 360*rotations_per_s
    degrees_per_ms = degrees_per_s/1e3

    window = init_window(windowsize=windowsize)
    set_opengl_params()
    clock = pygame.time.Clock()

    mouse_down = False
    pos = pygame.mouse.get_pos()
    prev_pos = pygame.mouse.get_pos()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            mouse_down = is_mouse_down(event, mouse_down)
            rotate_obj = should_obj_rotate(event, mouse_down)
            prev_pos = pos[:]
            pos = pygame.mouse.get_pos()

            if (rotate_obj):
                execute_mouse_controlled_rotation(octahedron, *pos, *prev_pos)
                    
        time = clock.tick()
        
        if rotate:
            rotate_view(octahedron, time*degrees_per_ms, *rotation_axis, use_opengl=use_opengl)
        
        update_screen(octahedron, cmap=blue_cmap, gl_mode=gl_mode)

def draw_cube(windowsize=(700,700), gl_mode=GL_QUADS, rotate=False, 
                    use_opengl=False, rotations_per_s = 5, rotation_axis=(0,0,1)):
    cube = [
        Face3D((-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)),
        Face3D((-1,-1,-1), (-1,1,-1), (1,1,-1), (1,-1,-1)),
        Face3D((1,-1,-1), (1,1,-1), (1,1,1), (1,-1,1)),
        Face3D((-1,-1,-1), (-1,-1,1), (-1,1,1), (-1,1,-1)),
        Face3D((-1,1,-1), (-1,1,1), (1,1,1), (1,1,-1)),
        Face3D((-1,-1,-1), (1,-1,-1), (1,-1,1), (-1,-1,1)),
    ]

    degrees_per_s = 360*rotations_per_s
    degrees_per_ms = degrees_per_s/1e3

    window = init_window(windowsize=windowsize)
    set_opengl_params()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                    
        time = clock.tick()
        
        if rotate:
            rotate_view(cube, time*degrees_per_ms, *rotation_axis, use_opengl=use_opengl)
        
        update_screen(cube, cmap=blue_cmap, gl_mode=gl_mode)