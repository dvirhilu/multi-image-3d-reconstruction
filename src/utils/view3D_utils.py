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
import utils.linalg_utils as linalg
from utils.face_3d import Face3D

def init_window(windowsize=(400,400)):
    '''
    @brief  initialize pygame window

    @param windowsize   The size of the generated window
    @return             The generated pygame window
    '''
    pygame.init()
    window = pygame.display.set_mode(windowsize, DOUBLEBUF|OPENGL)
    return window

def set_opengl_params(perspective_distance=20):
    '''
    @brief  Initialize OpenGL parameters

    @param perspective_distance The distance between origin and observation 
                                point in cm
    '''
    gluPerspective(45, 1, 0.1, 50.0)
    glTranslatef(0.0,0.0, -perspective_distance)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glCullFace(GL_BACK) 

def is_mouse_down(event, mouse_down):
    '''
    @brief  Determines whether mouse was pressed

    @param event        pygame event 
    @param mouse_down   is the mouse already pressed
    @return             Whether the mouse is currently pressed
    '''
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        return True
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        return False
    else:
        return mouse_down

def mouse_rotation_requested(event, mouse_down):
    '''
    @brief  Was the mouse moved while pressed by the used

    @param event        pygame event 
    @param mouse_down   is the mouse already pressed
    @return             Whether the mouse was moved while pressed
    '''
    return event.type == pygame.MOUSEMOTION and mouse_down

def execute_mouse_controlled_point_rotation(points, x, y, x_prev, y_prev, use_opengl=False):
    '''
    @brief  Rotate the points based on mouse movement 

    @param points       The point cloud 
    @param x            Current mouse x position
    @param y            Current mouse y position
    @param x_prev       Previous mouse x position
    @param y_prev       Previous mouse y position
    @param use_opengl   Use OpenGL or custom rotation method
    @return             New rotated points if opengl not used. Original points 
                        if opengl used (opengl rotates view not points)
    '''
    # mouse coordinates increase y in downward direction
    displacement_vec = np.array([x-x_prev, -(y-y_prev), 0])
    if linalg.norm(displacement_vec) == 0:
        return points
    rotation_axis = np.cross(displacement_vec, np.array([0,0,-1]))
    rotation_angle = 0.3*linalg.norm(displacement_vec)
    
    if use_opengl:
        glRotatef(rotation_angle, *rotation_axis)
        return points
    else:
        angle = np.pi/180 * rotation_angle
        rotated_points = []
        for point in points:
            rotated_point = linalg.rotate_vec(point, angle, rotation_axis)
            rotated_points.append(rotated_point)

        return rotated_points
    
def update_points_on_screen(points):
    '''
    @brief  Update the screen with new point coordinates

    @param points       The point cloud 
    '''
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glBegin(GL_POINTS)

    for point in points:
        glVertex3fv(point)

    glEnd()
    pygame.display.flip()

def view_point_cloud_interactively(points, windowsize=(700,700)):
    '''
    @brief  Creates a pygame window that allows to interactively view and 
            rotate the point cloud

    @param points       The point cloud
    @param windowsize   The size of the created pygame window
    '''
    # set window and graphics related parameters
    window = init_window(windowsize=windowsize)
    set_opengl_params()
    clock = pygame.time.Clock()

    # parameters for mouse controlled rotation
    mouse_down = False
    pos = pygame.mouse.get_pos()
    prev_pos = pygame.mouse.get_pos()

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

            # rotate the point cloud if mouse controlled rotation was srequested
            if (rotate_obj):
                points = execute_mouse_controlled_point_rotation(points, *pos, *prev_pos)
        
        update_points_on_screen(points)
