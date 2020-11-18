import cv2
import numpy as np
from imutils import rotate_bound
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from matplotlib import cm
import utils.plt_utils as plt_utils
from utils.linalg_utils import get_euclidean_distance, get_vec_cos_angle, cart2d_2_pol
from functools import reduce
import utils.file_io_utils as file_io_utils

def gaussian_filter(cutoff, rows, cols, center=True):
    u = np.linspace(-0.5, 0.5, rows)
    v = np.linspace(-0.5, 0.5, cols)

    H = np.zeros( (rows, cols) )
    for i in range(rows):
        for j in range(cols):
            r_squared = u[i]**2 + v[j]**2
            exponent = -0.5*r_squared/cutoff**2
            H[i,j] = np.e ** exponent

    if not center:
        return ifftshift(H)

    return H

def apply_gaussian_blur(image, fourier=False, sigma=3, mask_len=9, cutoff=0.25):
    
    if fourier:
        # transform image to frequency domain and center
        imfft = fftshift(fft2(image))

        # get image dimensions
        rows, cols = imfft.shape

        # generate gaussian filter
        H = gaussian_filter(cutoff, rows, cols)

        # apply gaussian filter in frequency domain
        imfft_filtered = imfft * H

        # plt_utils.show_images(logmag(H), logmag(imfft), logmag(imfft_filtered))
        
        # transform back and output
        return np.abs(ifft2(ifftshift(imfft_filtered)))
    else:
        h = gaussian_mask(mask_len, sigma)

        return convolve2d(image, h, 'same')

def gaussian_mask(length, sigma):
    x = gaussian(length, sigma)
    return np.outer(x, x)

def fx_mask():
    return np.array([
        [0, 0, 0],
        [1, 0, -1],
        [0, 0, 0]
    ])

def fy_mask():
    return np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, -1, 0]
    ])

def sobelx_mask():
    return np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

def sobely_mask():
    return np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

def fxx_mask():
    fx = fx_mask()

    return convolve2d(fx, fx)

def fyy_mask():
    fy = fy_mask()

    return convolve2d(fy, fy)

def fxy_mask():
    fx = fx_mask()
    fy = fy_mask()

    return convolve2d(fx, fy)

def sobelxx_mask():
    fx = sobelx_mask()

    return convolve2d(fx, fx)

def sobelyy_mask():
    fy = sobely_mask()

    return convolve2d(fy, fy)

def sobelxy_mask():
    fx = sobelx_mask()
    fy = sobely_mask()

    return convolve2d(fx, fy)

def apply_hessian_corner_mask_fast(image, c, sobel=False):
    if sobel:
        fxx = sobelxx_mask()
        fyy = sobelyy_mask()
        fxy = sobelxy_mask()
    else:
        fxx = fxx_mask()
        fyy = fyy_mask()
        fxy = fxy_mask()
        

    image_xx = convolve2d(image, fxx, mode='same')
    image_yy = convolve2d(image, fyy, mode='same')
    image_xy = convolve2d(image, fxy, mode='same')

    # find eigenvalues of hessian matrix
    descriminant = np.sqrt(
        (image_xx - image_yy)**2 + 4*image_xy**2
    )
    lambda1 = 0.5*(image_xx + image_xy + descriminant)
    lambda2 = 0.5*(image_xx + image_xy - descriminant)

    # filter pixels to be 1 for a detected saddle point, 0 otherwise
    # Saddle points occur when lambda1>0 and lambda2<0
    # to filter noise, use a threshold slightly above 0
    threshold = c*np.amax(lambda1)
    outimage = (lambda1 > threshold) & (lambda2 < -threshold)

    # output the number of corners detected
    # print(np.sum(outimage))

    return outimage

def apply_hessian_corner_mask(image, windowsize=3, sobel=False):
    if sobel:
        fxx = sobelxx_mask()
        fyy = sobelyy_mask()
        fxy = sobelxy_mask()
    else:
        fxx = fxx_mask()
        fyy = fyy_mask()
        fxy = fxy_mask()

    image_xx = convolve2d(image, fxx, mode='same')
    image_yy = convolve2d(image, fyy, mode='same')
    image_xy = convolve2d(image, fxy, mode='same')
    
    determinant = image_xx*image_xy - image_xy**2

    plt_utils.show_images(determinant, image_xx, image_yy, image_xy)

    # filter to only keep pixels that are local minimum within a window of windowsize and have determinant <= 0
    rows, cols = determinant.shape
    delta = windowsize // 2
    outimage = np.zeros((rows, cols), dtype=bool)
    for i in range(delta, rows-delta):
        for j in range(delta, cols-delta):
            local_min = determinant[i, j] < np.amin(determinant[(i+1):(i+delta+1), (j-delta):(j+delta+1)])
            local_min = (determinant[i, j] < np.amin(determinant[i, (j-delta):j])) & local_min
            local_min = (determinant[i, j] < np.amin(determinant[i, (j+1):(j+delta+1)])) & local_min
            local_min = (determinant[i, j] < np.amin(determinant[(i-delta):i, (j-delta):(j+delta+1)])) & local_min
            nonpositive_element = determinant[i,j] <= 0

            outimage[i, j] = local_min & nonpositive_element

    # output the number of corners detected
    print(np.sum(outimage))

    return outimage

def apply_harris_corner_filter(image, windowsize=10, sobel_size=3, k=0.04, use_cv=True, p=0.1):
    if use_cv:
        image = np.float32(image)
        dst = cv2.cornerHarris(image,windowsize, sobel_size, k)

        # Threshold for an optimal value, it may vary depending on the image.
        corners = dst > p*dst.max()

        return corners

def squish_cluster_2_centroid_filter(corner_mask):
    rows, cols = corner_mask.shape[:2]

    for i in range(1,rows-1):
        for j in range(1, cols-1):
            if corner_mask[i, j]:
                # get cluster around discovered point
                (x1, x2, y1, y2) = find_cluster(corner_mask, i, i+1, j, j+1)
                # extract window containing cluster
                window = corner_mask[x1:x2, y1:y2]
                # find cluster centroid
                (centroidx, centroidy) = find_centroid(window)

                # create new window with only centroid as 1
                new_window = np.zeros(window.shape, dtype=bool)
                centroidx = int(round(centroidx))
                centroidy = int(round(centroidy))
                new_window[centroidx, centroidy] = True

                # update corner mask cluster to centroid only
                corner_mask[x1:x2, y1:y2] = new_window

    
    return corner_mask

def find_centroid(window):
    rows, cols = window.shape[:2]

    total_mass = np.sum(window)
    x_centroid = 0
    y_centroid = 0
    for i in range(rows):
        for j in range(cols):
            mass = window[i,j]
            x_centroid += i*mass / total_mass
            y_centroid += j*mass / total_mass
    
    return(x_centroid, y_centroid)


def find_cluster(corner_mask, xstart, xstop, ystart, ystop):
    left_neighbours = corner_mask[xstart-1, ystart-1:ystop+1]
    right_neighbours = corner_mask[xstop+1, ystart-1:ystop+1]
    top_neighbours = corner_mask[xstart-1:xstop+1, ystart-1]
    bottom_neighbours = corner_mask[xstart-1:xstop+1, ystop+1]

    # if neighbours on any side include elements, expand to that side
    inc_left    = reduce(lambda a,b: a or b, left_neighbours)    
    inc_right   = reduce(lambda a,b: a or b, right_neighbours)    
    inc_top     = reduce(lambda a,b: a or b, top_neighbours)    
    inc_bottom  = reduce(lambda a,b: a or b, bottom_neighbours)

    # flag that a resize is needed
    increase_window_size = inc_left and inc_right and inc_top and inc_bottom

    if increase_window_size:
        # calcualte new expanded window dimensions based on where expansion is needed
        xstart  = xstart-1  if inc_left     else xstart
        xstop   = xstop+1   if inc_right    else xstop
        ystart  = ystart-1  if inc_top      else ystart
        ystop   = ystop+1   if inc_bottom    else ystop

        # call function recuresively until window does not need to resize
        return find_cluster(corner_mask, xstart, xstop, ystart, ystop)
    
    else:
        return (xstart, xstop, ystart, ystop)
    
    # areas_around_window = np.array([
    #     corner_mask[xstart-1:xstop+1, ystart-1],
    #     corner_mask[xstart-1:xstop+1, ystop+1],
    #     corner_mask[xstart-1, ystart-1:ystop+1],
    #     corner_mask[xstop+1, ystart-1:ystop+1]
    # ]).flatten()

    # # check if any of the neighbours are 1. If so, window size should increase
    # increase_window_size = reduce(lambda a, b: a or b, areas_around_window)

    # if increase_window_size:
    #     return find_cluster(corner_mask, xstart-1, xstop+1, ystart-1, ystop+1)
    # else:
    #     return (xstart, xstop, ystart, ystop)


def filter_corner_centrosymmetry(image, corner_mask, circular_mask_radius, cutoff_ratio):
    I_masks = [
        partially_averaging_circular_mask(circular_mask_radius, i*360/8, (i+1)*360/8)
        for i in range(8)
    ]

    # apply partial circular masks to find average in different sections
    I = [
        convolve2d(image, kernel, 'same')
        for kernel in I_masks
    ]

    # average intensity differences between areas in circular mask
    D1 = np.abs(I[0] - I[4]) # D1 = |I1 - I5|
    D2 = np.abs(I[2] - I[6]) # D2 = |I3 - I7|
    D3 = np.abs(I[0] + I[4] - I[2] - I[6])/2 # D3 = |I1 + I5 - I3 - I7|/2
    D4 = np.abs(I[1] - I[5]) # D4 = |I2 - I6|
    D5 = np.abs(I[3] - I[7]) # D5 = |I4 - I8|
    D6 = np.abs(I[1] + I[5] - I[3] - I[7])/2 # D6 = |I2 + I6 - I4 - I8|/2

    centosymmetry_criteria_1_mask = (D1 < cutoff_ratio*D3) | (D2 < cutoff_ratio*D3)
    centosymmetry_criteria_2_mask = (D4 < cutoff_ratio*D6) | (D5 < cutoff_ratio*D6)

    # keep corner if either centosymmetry criteria pass
    outmask = corner_mask & (centosymmetry_criteria_1_mask|centosymmetry_criteria_2_mask)

    # output how many corners were detected
    print(np.sum(outmask))

    return outmask


def partially_averaging_circular_mask(radius, phi_start, phi_end):
    def in_averaging_regions(i, j, radius, phi_start, phi_end):
        r, phi = cart2d_2_pol(i-radius, radius-j) # y coordinate is flipped
        inside_circle  = r <= radius
        # account for overlap
        if phi_end > phi_start:
            inside_angular_range = phi_start <= phi < phi_end
        else:
            inside_angular_range = phi < phi_end or phi > phi_start

        return inside_circle and inside_angular_range

    # change to radian angle between -pi and pi
    phi_start = ((phi_start + 180)%360 - 180)*np.pi/180
    phi_end = ((phi_end + 180)%360 - 180)*np.pi/180

    # insert ones in averaging slice, zeros otherwise
    mask = np.array([
        [
            1 if in_averaging_regions(i, j, radius, phi_start, phi_end) else 0
            for i in range(2*radius)
        ]
        for j in range(2*radius)        
    ])

    # normalize and return
    return mask / np.sum(mask)

def filter_corner_distance(corner_mask, distance_threshold, num_neighbours):
    corner_indices = get_true_indices(corner_mask)

    nearest_neighbours = [
        find_k_nearest_neighbours(index_pair, corner_indices, num_neighbours)
        for index_pair in corner_indices
    ]

    # create copy of amsk to not mutate input
    outmask = corner_mask[:]
    # eliminate corners not matching the distance criteria
    for index_pair, neighbours in zip(corner_indices, nearest_neighbours):
        distances_above_threshold = [
            get_euclidean_distance(index_pair, neighbour) > distance_threshold
            for neighbour in neighbours
        ]

        above_thresh_exists = reduce(lambda a, b: a or b, distances_above_threshold)

        if above_thresh_exists:
            outmask[index_pair[0], index_pair[1]] = False

    return outmask


def filter_corner_angle(corner_mask, cos_threshold):
    corner_indices = get_true_indices(corner_mask)

    nearest_2_neighbours = [
        find_k_nearest_neighbours(index_pair, corner_indices, 2)
        for index_pair in corner_indices
    ]

    # create copy of amsk to not mutate input
    outmask = corner_mask[:]
    # eliminate corners not matching the angle criteria
    for index_pair, neighbours in zip(corner_indices, nearest_2_neighbours):
        vec1 = index_pair - neighbours[0]
        vec2 = index_pair - neighbours[1]

        cos_theta = get_vec_cos_angle(vec1, vec2)

        if cos_theta < cos_threshold:
            outmask[index_pair[0], index_pair[1]] = False

    return outmask

def get_true_indices(logical_mask):
    rows, cols = logical_mask.shape
    return [
        np.array([i, j])
        for i in range(rows)
        for j in range(cols)
        if logical_mask[i,j]
    ]

def find_k_nearest_neighbours(point, neighbours, k):
    # copy list to not mutate input
    neighbour_list = neighbours[:]
    
    # sort index list based on distance
    neighbour_list.sort(key=lambda neighbour: get_euclidean_distance(point, neighbour))

    # check if point is included in the neighbour list
    point_in_neighbour_list = any(
        np.array_equal(point, neighbour) 
        for neighbour in neighbour_list
    )

    # return the k nearest neighbours excluding the point itself
    return neighbour_list[1:k+1] if point_in_neighbour_list else neighbour_list[:k]

def get_nearest_neighbour_distance_distribution(corner_mask):
    corner_indices = get_true_indices(corner_mask)

    nearest_neighbours = [
        find_k_nearest_neighbours(index_pair, corner_indices, 1)
        for index_pair in corner_indices
    ]

    return [
        get_euclidean_distance(index_pair, neighbour)
        for (index_pair, neighbour) in zip(corner_indices, nearest_neighbours)
    ]

def logmag(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i,j] = 1e-300 if im[i,j]==0 else im[i,j]
    return np.log(np.abs(im))

def get_undistored_k_matrix(image, k, d):
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 1, (w,h))

    return newcameramtx, roi

def undistort(object_name, calib_name):
    images = io.load_object_images(object_name)
    k, d = io.load_calib_coefficients(calib_name)

    for (image, i) in zip(images, range(len(images))):            
        newcameramtx, roi = get_undistored_k_matrix(image, k, d)

        # undistort
        undst = cv2.undistort(image, k, d, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        undst = undst[y:y+h, x:x+w]

        # write new camera calibration info
        
        cv2.imwrite("images/objects/" + object_name + "_undistorted/image" + str(i) + ".png", undst)

if __name__=="__main__":

    images = file_io_utils.load_object_images("monkey_thing")[:2]
    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]
    plt_utils.show_images(*images)

    # # apply gaussian blur
    # sigma = 2
    # mask_length = 10
    # cutoff = 0.2
    # images = [
    #     # apply_gaussian_blur(image, fourier=True, cutoff=cutoff)
    #     apply_gaussian_blur(image, fourier=False, sigma=sigma, mask_len=mask_length)
    #     for image in images
    # ]
    
    # # show blurred images
    # plt_utils.show_images(*images)

    windowsize = 10
    sobel_size = 3
    k = 0.04
    p = 0.1
    corners = [
        apply_harris_corner_filter(image, windowsize=windowsize, sobel_size=sobel_size, k=k, p=p)
        for image in images
    ]

    plt_utils.plot_corner_points(images, corners)

    corners = [
        squish_cluster_2_centroid_filter(corner)
        for corner in corners
    ]

    plt_utils.plot_corner_points(images, corners)
    
    plt.show()

    # windowsize=5
    # # apply hessian determinate mask
    # corners = [
    #     apply_hessian_corner_mask(image, windowsize=windowsize, sobel=False)
    #     for image in images
    # ]

    # print(np.where(corners[0]))

    # # show hessian'd images
    # plt_utils.plot_corner_points(images, corners)
    # plt_utils.show_images(*corners)

    # distances = get_nearest_neighbour_distance_distribution(im)
    # distances_rot = get_nearest_neighbour_distance_distribution(im_rot)
    # distances_aff = get_nearest_neighbour_distance_distribution(im_aff)

    # hist_titles = [
    #     title + "Hess Distance Distribution"
    #     for title in im_titles_orig
    # ]

    # plt_utils.plt_histograms(distances, distances_rot, distances_aff, titles=hist_titles)

    # # filter for centrosymmetry property
    # radius = 20
    # p = 0.2
    # im = filter_corner_centrosymmetry(im_orig, im, radius, p)
    # im_rot = filter_corner_centrosymmetry(im_rot_orig, im_rot, radius, p)
    # im_aff = filter_corner_centrosymmetry(im_aff_orig, im_aff, radius, p)

    # im_titles = [
    #     title + " Sym Filter"
    #     for title in im_titles_orig
    # ]

    # # show centrosymmetry filtered image
    # plt_utils.show_images(im, im_rot, im_aff, titles=im_titles)

    # # filter for distance property
    # max_dist = 80
    # im = filter_corner_distance(im, max_dist, 3)
    # im_rot = filter_corner_distance(im_rot, max_dist, 3)
    # im_aff = filter_corner_distance(im_aff, max_dist, 3)

    # im_titles = [
    #     title + " Dist Filter"
    #     for title in im_titles_orig
    # ]

    # # show distance filtered image
    # plt_utils.show_images(im, im_rot, im_aff, titles=im_titles)

    # # filter for angle property
    # num_iter = 3
    # cos_thresh = 0.87
    # for i in range(num_iter):
    #     im = filter_corner_angle(im, cos_thresh)
    #     im_rot = filter_corner_angle(im_rot, cos_thresh)
    #     im_aff = filter_corner_angle(im_aff, cos_thresh)

    # im_titles = [
    #     title + " Dist Filter"
    #     for title in im_titles_orig
    # ]

    # # show distance filtered image
    # plt_utils.show_images(im, im_rot, im_aff, titles=im_titles)