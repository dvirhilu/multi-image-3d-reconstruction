%==========================================================================
% Multi-Image 3D Reconstruction
% Save Camera Coefficients
% 
% After a cameraParams object was created by MATLAB, this script is used to
% save the coefficients to a yaml file. This allows to interact with the
% camera coefficients in openCV on Python
% 
% The script outputs the camera parameters to 
% calib_params/<camera_name>.yml
% 
% Required: cameraParams object generated using find_camera_params.m
% 
% Author:   Dvir Hilu
% Date:     11/11/2020
%==========================================================================

% ===========================
% Script Parameters
% ===========================
calib_name = "SamsungGalaxyA8";

% ===========================
% Save Coefficients to File
% ===========================
outfile = fopen(append("calib_params/", calib_name, ".yaml"),'w');

k = cameraParams.IntrinsicMatrix';
d = cameraParams.RadialDistortion;
p = cameraParams.TangentialDistortion;

fprintf(outfile,'k:\n  rows: 3\n  cols: 3\n  data: [%9f %9f %9f %9f %9f %9f %9f %9f %9f]\n', ...
    k(1,1), k(1,2), k(1,3), ...
    k(2,1), k(2,2), k(2,3), ...
    k(3,1), k(3,2), k(3,3));

fprintf(outfile,'d:\n  rows: 1\n  cols: 2\n  data: [%9f %9f %9f %9f %9f]\n', ...
    d(1,1), d(1,2), p(1,1), p(1,2), d(1,3));

fclose(outfile);