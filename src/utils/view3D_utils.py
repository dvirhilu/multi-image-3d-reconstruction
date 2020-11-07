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
from utils import linalg_utils
from utils.common_types import *

blue_cmap = cm.get_cmap("Blues")

def get_shade_val(face, light_dir=np.array([1,1,1])):
    unit_normal = linalg_utils.unit_vect(face.normal_vector)
    unit_light_dir = linalg_utils.unit_vect(light_dir)
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

def mouse_rotation_requested(event, mouse_down):
    return event.type == pygame.MOUSEMOTION and mouse_down

def auto_rotation_requested(event, auto_rotating):
    if (event.type == KEYDOWN):
        # toggle auto_rotating when r is pressed
        if event.key == pygame.K_r:
            return not auto_rotating

    return auto_rotating

def get_auto_rotation_axis(event, rotation_direction):
    if (event.type == KEYDOWN):
        if event.key == pygame.K_s:
            return [np.random.uniform(-1,1) for i in range(3)]

    return rotation_direction

def execute_mouse_controlled_rotation(face_list, x, y, x_prev, y_prev):
    # mouse coordinates increase y in downward direction
    displacement_vec = np.array([x-x_prev, -(y-y_prev), 0])
    if linalg_utils.norm(displacement_vec) == 0:
        return
    rotation_axis = np.cross(displacement_vec, np.array([0,0,-1]))
    rotation_angle = 0.3*linalg_utils.norm(displacement_vec)
    
    rotate_view(face_list, rotation_angle, *rotation_axis)

def execute_auto_rotation(face_list, rotation_axis, time):
    degrees_per_ms = 36e-3
    rotate_view(face_list, time*degrees_per_ms, *rotation_axis)

def view_object_interactively(face_list, gl_mode=GL_TRIANGLES, windowsize=(700,700), cmap=blue_cmap):
    # set window and graphics related parameters
    window = init_window(windowsize=windowsize)
    set_opengl_params()
    clock = pygame.time.Clock()

    # parameters for mouse controlled rotation
    mouse_down = False
    pos = pygame.mouse.get_pos()
    prev_pos = pygame.mouse.get_pos()

    # parameters for automatic rotation
    auto_rotation_axis = [np.random.uniform(-1,1) for i in range(3)]
    auto_rotate = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # handle mouse controlled rotation
            mouse_down = is_mouse_down(event, mouse_down)
            rotate_obj = mouse_rotation_requested(event, mouse_down)
            prev_pos = pos[:]
            pos = pygame.mouse.get_pos()

            if (rotate_obj):
                execute_mouse_controlled_rotation(face_list, *pos, *prev_pos)

            # # handle auto-rotation
            # auto_rotate = auto_rotation_requested(event, auto_rotate)
            # time = clock.tick()
            # auto_rotation_axis = get_auto_rotation_axis(event, auto_rotation_axis)

            # if (auto_rotate):
            #     execute_auto_rotation(face_list, auto_rotation_axis, time)
        
        update_screen(face_list, cmap=cmap, gl_mode=gl_mode)
        print(clock.get_fps())