%==========================================================================
% Multi-Image 3D Reconstruction
% Automatic Camera Calibration
% 
% This script, given a set of 10 more images of a chessboard pattern,
% extracts the distortion parameters of the camera lens as well as the
% intrinsic parameters (focal length, principal point, skew coefficient).
% 
% The script takes in images from images/calibration/<camera_name>/*
% and outputs the camera parameters to calib_params/<camera_name>.yml
% 
% Required: MATLAB Computer Vision Toolbox
% 
% Author:   Dvir Hilu
% Date:     06/10/2020
%==========================================================================
close all;
clear all;

% ===========================
% Script Parameters
% ===========================
camera_name = "SamsungGalaxyA8";
square_size = 1.9; % cm

% ===========================
% Launch Camera Calibration Tool
% ===========================
cameraCalibrator(append("../images/calibration/", camera_name, "/"), square_size);