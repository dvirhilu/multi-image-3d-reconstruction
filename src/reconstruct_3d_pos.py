import numpy as np
import utils.linalg_utils as linalg
import cv2
from sift import SIFTFeature
from itertools import combinations

def create_ls_matrix(proj_mats, image_points):
    '''
    @brief  Generates the homogeneous least squares matrix from the feature 
            image points and their corresponding projection matrices

    @param proj_mats        The projection matrices for each image, ordered by 
                            the image index
    @param image_points     A list of coordinates of the matched features, 
                            also ordered by image index
    @return                 The homogeneous least squares matrix
    '''
    # generate rows for all x image point coordinate equations
    x_cross_prod_eqns = np.array([
        point[0]*P[2, :].T - P[0, :].T
        for (point, P) in zip(image_points, proj_mats)
    ])

    # generate rows for all y image point coordinate equations
    y_cross_prod_eqns = np.array([
        point[1]*P[2, :].T - P[1, :].T
        for (point, P) in zip(image_points, proj_mats)
    ])

    return np.concatenate((x_cross_prod_eqns, y_cross_prod_eqns), axis=0)

def reconstruct_3D_points(feature_groups, proj_mats):
    '''
    @brief  Generates a point cloud from the feature groups by using the least 
            squares solution to a homogeneous system

    @param feature_groups   A list of the feature groups used in the 
                            reconstruction.
    @param proj_mats        The projection matrices for each image, ordered by 
                            the image index
    @return                 A list of reconstructed points
    '''
    reconstructed_points = []
    for group in feature_groups:
        # exctract projection matrices corresponding to the features
        relevant_proj_mats = [
            proj_mats[feature.image_idx]
            for feature in group
        ]

        # extract feature image coordinates
        image_points = [
            feature.coordinates
            for feature in group
        ]

        # generate homogeneous least squares matrix
        ls_mat = create_ls_matrix(relevant_proj_mats, image_points)

        # solve linear least squares system
        world_point = linalg.solve_homogeneous_least_squares(ls_mat)

        # convert back from homogeneous coordinates
        world_point = world_point[0:3] / world_point[3]

        reconstructed_points.append(world_point)

    return reconstructed_points

def compute_mean_reprojection_error(reconstructed_point, feature_group, P_mats):
    '''
    @brief  Computes the of mean reprojection error of a single reconstructed 
            point

    @param reconstructed_point  The reconstructed point
    @param feature_groups       The feature group corresponding to the
                                reconstructed point
    @param P_mats               The projection matrices for each image, 
                                ordered by the image index
    @return                     The mean reprojection error of the 
                                reconstructed point
    '''
    total_error = 0
    for feature in feature_group:
        # compute reprojection
        P = P_mats[feature.image_idx]
        X = np.append(reconstructed_point, [1], axis=0).reshape(4,1)
        reprojection_homogeneous = P @ X
        reprojection = reprojection_homogeneous[:2, 0] / reprojection_homogeneous[2,0]

        # get original image point
        image_point = np.array(feature.coordinates)

        total_error += linalg.get_euclidean_distance(reprojection, image_point)

    return total_error / len(feature_group)

def compute_reprojection_error_distribution(reconstructed_points, feature_groups, P_mats):
    '''
    @brief  Computes the distribution of mean reprojection error for the
            reconstructed points

    @param reconstructed_points The orignal point cloud
    @param feature_groups       A list of the feature groups used in the
                                reconstruction. These should be in the same 
                                order as the reconstructed points the 
                                correspond to
    @param P_mats               The projection matrices for each image, 
                                ordered by the image index
    @return                     A list containing the mean reprojection error 
                                of each reconstructed point
    '''
    return [
        compute_mean_reprojection_error(point, group, P_mats)
        for (point, group) in zip(reconstructed_points, feature_groups)
    ]

def filter_reprojection_error(reconstructed_points, feature_groups, P_mats, reprojection_error_threshold=20):
    '''
    @brief  Filters the point cloud to remove points have high reprojection 
            error

    @param reconstructed_points         The orignal point cloud
    @param feature_groups               A list of the feature groups used in 
                                        the reconstruction. These should be in 
                                        the same order as the reconstructed 
                                        points the correspond to
    @param P_mats                       The projection matrices for each 
                                        image, ordered by the image index
    @param reprojection_error_threshold Points with a mean absolute 
                                        reprojection error of more than 
                                        reprojection_error_threshold are 
                                        discarded
    @return                             The new filtered point cloud
    '''
    # compute mean reprojection error for each reconstructed point
    reprojection_error = compute_reprojection_error_distribution(
        reconstructed_points, 
        feature_groups, 
        P_mats
    )

    # filter out points above MAE threshold
    return [
        reconstructed_points[i]
        for i in range(len(reconstructed_points))
        if reprojection_error[i] < reprojection_error_threshold
    ]

def filter_xyz_outliers(reconstructed_points, num_stdev=1, z_percentile=80):
    '''
    @brief  Filters the point cloud to remove points that are outliers in xyz 
            distributions

    @param reconstructed_points The orignal point cloud
    @param num_stdev            Points located beyond num_stdev in the x and y 
                                distributions are discarded
    @param z_percentile         Points located beyond the z_percentile in the 
                                z distribution are discarded
    @return                     The new filtered point cloud
    '''
    # filter points with z < 0
    reconstructed_points = [
        point
        for point in reconstructed_points
        if point[2] >= 0
    ]

    # compute point centroid
    centroid = linalg.compute_point_group_centroid(reconstructed_points)

    # get x and y distributions shifted by centroid
    x_dist = [
        point[0] - centroid[0]
        for point in reconstructed_points
    ]

    y_dist = [
        point[1] - centroid[1]
        for point in reconstructed_points
    ]

    # get z distribution (z is compared relative to z=0 not the centroid)
    z_dist = [
        point[2]
        for point in reconstructed_points
    ]

    # define cutoff values
    x_cutoff = np.std(x_dist) * num_stdev
    y_cutoff = np.std(y_dist) * num_stdev
    z_cutoff = np.percentile(z_dist, z_percentile)

    # only retain point if it is within all cutoffs
    return [
        reconstructed_points[i]
        for i in range(len(reconstructed_points))
        if  np.abs(x_dist[i]) < x_cutoff
        and np.abs(y_dist[i]) < y_cutoff
        and z_dist[i] < z_cutoff
    ]

def shift_points_to_centroid(points):
    '''
    @brief  Shifts point cloud so that the origin is at its centroid

    @param points       The original point cloud
    @return             The new shifted point cloud
    '''
    centroid = linalg.compute_point_group_centroid(points)
    return [
        point - centroid
        for point in points
    ]

def add_background_surface(points, num_stdev=1):
    '''
    @brief  Adds points in reconstructed point cloud to indicate where the 
            flat x-corner board is

    @param points       The original point cloud
    @param num_stdev    Extent of points. Indicates how many standard 
                        deviations in x and y distributions of original 
                        point cloud to draw the points for 
    @return             The updated point cloud
    '''
    x_dist = [
        point[0]
        for point in points
    ]

    y_dist = [
        point[1]
        for point in points
    ]

    meanx = np.mean(x_dist)
    stdevx = np.std(x_dist)
    meany = np.mean(y_dist)
    stdevy = np.std(y_dist)
    xmin = meanx-stdevx*num_stdev
    xmax = meanx+stdevx*num_stdev
    ymin = meany-stdevy*num_stdev
    ymax = meany+stdevy*num_stdev

    background_surface_points = [
        np.array([x, y, 0])
        for x in np.linspace(xmin, xmax, int(10*num_stdev))
        for y in np.linspace(ymin, ymax, int(10*num_stdev))
    ]

    return points + background_surface_points