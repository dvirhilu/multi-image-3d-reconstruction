import cv2
import numpy as np
from imutils import rotate_bound
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from matplotlib import cm
from utils.im_view_utils import show_images
from utils.linalg_utils import get_euclidean_distance, get_vec_cos_angle, cart2d_2_pol
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

        # show_images(logmag(H), logmag(imfft), logmag(imfft_filtered))
        
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
    # print(np.sum(outimage))

    return outimage

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
    im_orig = cv2.imread("../../camera_calibration/calib_images/chess_pattern.png")
    im_orig = cv2.cvtColor(im_orig ,cv2.COLOR_BGR2GRAY)
    rows, cols = im_orig.shape

    im_orig = add_noise(im_orig, 50)

    im_rot_orig = rotate_bound(im_orig, 45)
  
    pts1 = np.float32([[50, 50], 
                       [200, 50],
                       [50, 200]]) 
    
    pts2 = np.float32([[-5, -50], 
                       [100, 25],  
                       [50, 125]]) 
    
    M = cv2.getAffineTransform(pts1, pts2) 
    im_aff_orig = cv2.warpAffine(im_orig, M, (cols, rows))

    im_titles_orig = ["Original Image", "Rotated Image", "Affined Image"]

    # show original images
    show_images(im_orig, im_rot_orig, im_aff_orig, titles=im_titles_orig)

    # apply gaussian blur
    sigma = 20
    mask_length = 60
    im = apply_gaussian_blur(im_orig, fourier=False, sigma=sigma, mask_len=9)
    im_rot = apply_gaussian_blur(im_rot_orig, fourier=False, sigma=sigma, mask_len=9)
    im_aff = apply_gaussian_blur(im_aff_orig, fourier=False, sigma=sigma, mask_len=9)

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
    # show_images(im, im_rot, im_aff, titles=im_titles)

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
    # show_images(im, im_rot, im_aff, titles=im_titles)

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
    # show_images(im, im_rot, im_aff, titles=im_titles)

    plt.show()