import cv2
import numpy as np
from imutils import rotate_bound
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib import cm
from utils.im_view_utils import show_images
from utils.linalg_utils import get_euclidean_distance, get_vec_angle, cart2d_2_pol
from functools import reduce

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

def apply_gaussian_filter(image, cutoff):
    # transform image to frequency domain and center
    imfft = fftshift(fft2(image))

    # get image dimensions
    rows, cols = imfft.shape

    # generate gaussian filter
    H = gaussian_filter(cutoff, rows, cols)

    # apply gaussian filter in frequency domain
    imfft_filtered = imfft * H

    # show_images(logmag(H), logmag(imfft), logmag(imfft_filtered))
    
    # transform back and output
    return np.abs(ifft2(ifftshift(imfft_filtered)))

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

def apply_hessian_corner_mask(image, c):
    fxx = fxx_mask()
    fyy = fyy_mask()
    fxy = fxy_mask()

    image_xx = convolve2d(image, fxx, mode='same')
    image_yy = convolve2d(image, fyy, mode='same')
    image_xy = convolve2d(image, fxy, mode='same')
    
    show_images(image_xx, image_yy, image_xy)

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
    print(np.sum(outimage))

    return outimage

def apply_hessian_corner_mask2(image, c):
    fxx = fxx_mask()
    fyy = fyy_mask()
    fxy = fxy_mask()

    image_xx = convolve2d(image, fxx, mode='same')
    image_yy = convolve2d(image, fyy, mode='same')
    image_xy = convolve2d(image, fxy, mode='same')

    # find eigenvalues of hessian matrix
    S = image_xx*image_yy - image_xy**2

    outimage = S < 0

    # output the number of corners detected
    print(np.sum(outimage))

    return outimage

def filter_corner_centrosymmetry(image, corner_mask):
    pass

def partially_averaging_circular_mask(radius, phi_start, phi_end):
    def in_averaging_regions(i, j, radius, phi_start, phi_end):
        r, phi = cart2d_2_pol(i-radius, j-radius)
        phi = phi*180/np.pi
        inside_circle  = r <= radius
        inside_angular_range = phi_start <= phi < phi_end

        return inside_circle and inside_angular_range

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
    rows, cols = corner_mask.shape
    corner_indices = [
        np.array([i, j])
        for i in range(rows)
        for j in range(cols)
        if corner_mask[i,j]
    ]

    nearest_2_neighbours = [
        find_k_nearest_neighbours(index_pair, corner_indices, num_neighbours)
        for index_pair in corner_indices
    ]

    # create copy of amsk to not mutate input
    outmask = corner_mask[:]
    # eliminate corners not matching the distance criteria
    for index_pair, neighbours in zip(corner_indices, nearest_2_neighbours):
        distances_above_threshold = [
            get_euclidean_distance(index_pair, neighbour) > distance_threshold
            for neighbour in neighbours
        ]

        above_thresh_exists = reduce(lambda a, b: a or b, distances_above_threshold)

        if above_thresh_exists:
            outmask[index_pair[0], index_pair[1]] = False

    return outmask


def filter_corner_angle(corner_mask, angle_threshold):
    rows, cols = corner_mask.shape
    corner_indices = [
        np.array([i, j])
        for i in range(rows)
        for j in range(cols)
        if corner_mask[i,j]
    ]

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

        angle = get_vec_angle(vec1, vec2)

        if np.abs(angle) < angle_threshold:
            outmask[index_pair[0], index_pair[1]] = False

    return outmask

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


def add_noise(im, sigma):
    rows, cols = im.shape

    for i in range(rows):
        for j in range(cols):
            x = i - rows/2
            y = j - cols/2
            r = np.sqrt(x**2 + y**2)
            should_swap = r < 40

            if should_swap:
                im[i][j] = 150
    
    return im

def logmag(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i,j] = 1e-300 if im[i,j]==0 else im[i,j]
    return np.log(np.abs(im))

if __name__=="__main__":
    # load chessboard image
    im = cv2.imread("../../camera_calibration/calib_images/chess_pattern.png")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rows, cols = im.shape

    im = add_noise(im, 50)

    im_rot = rotate_bound(im, 45)
  
    pts1 = np.float32([[50, 50], 
                       [200, 50],
                       [50, 200]]) 
    
    pts2 = np.float32([[-5, -50], 
                       [100, 25],  
                       [50, 125]]) 
    
    M = cv2.getAffineTransform(pts1, pts2) 
    im_aff = cv2.warpAffine(im, M, (cols, rows))

    im_titles_orig = ["Original Image", "Rotated Image", "Affined Image"]

    # show original images
    show_images(im, im_rot, im_aff, titles=im_titles_orig)

    # apply gaussian blur
    gaussian_cutoff = 0.2
    im = apply_gaussian_filter(im, gaussian_cutoff)
    im_rot = apply_gaussian_filter(im_rot, gaussian_cutoff)
    im_aff = apply_gaussian_filter(im_aff, gaussian_cutoff)

    im_titles = [
        title + " Blurred"
        for title in im_titles_orig
    ]
    
    # show blurred images
    show_images(im, im_rot, im_aff, titles=im_titles)

    threshold_const = 0.05
    # apply hessian determinate mask
    im = apply_hessian_corner_mask(im, threshold_const)
    im_rot = apply_hessian_corner_mask(im_rot, threshold_const)
    im_aff = apply_hessian_corner_mask(im_aff, threshold_const)

    im_titles = [
        title + " Hess Corners"
        for title in im_titles_orig
    ]

    # show hessian'd images
    show_images(im, im_rot, im_aff, titles=im_titles)

    plt.show()