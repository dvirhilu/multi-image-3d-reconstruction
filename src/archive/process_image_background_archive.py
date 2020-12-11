from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
import numpy as np
import utils.plt_utils as plt_utils

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
    # print(np.sum(outimage))

    return outimage

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

def logmag(im):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i,j] = 1e-300 if im[i,j]==0 else im[i,j]
    return np.log(np.abs(im))

def get_nearest_neighbour_distance_distribution(corner_mask):
    x = np.where(corner_mask)[1]
    y = np.where(corner_mask)[0]

    corner_indices = [
        np.array([y_val, x_val])
        for (y_val, x_val) in zip(y,x)
    ]

    nearest_neighbours = [
        find_k_nearest_neighbours(index_pair, corner_indices, 1)
        for index_pair in corner_indices
    ]

    return [
        linalg.get_euclidean_distance(index_pair, neighbour)
        for (index_pair, neighbour) in zip(corner_indices, nearest_neighbours)
    ]

def find_threshold_params(distances):
    num_elements = len(distances)

    # generate histogram with bin steps of 5 pixels
    num_bins = int((max(distances) - min(distances))/5)
    hist, bin_edges = np.histogram(distances, bins=num_bins)

    max_index = np.argmax(hist)

    r = 0
    for i in range(int(num_bins/2)):
        num_elements_in_window = np.sum(hist[max_index-i:max_index+i])
        power_percentage = num_elements_in_window / num_elements
        if power_percentage > 0.8:
            r = i
            break

    # remove elements not in 80% window
    window = np.array([
        x 
        for x in distances 
        if bin_edges[max_index-r] <= x <= bin_edges[max_index+1]
    ])

    # calculate amin and amax
    mu = np.average(window)
    sigma = np.std(window)
    amin = mu - 3*sigma
    amax = mu + 3*sigma

    # calcualte threshold values
    r = int(round(0.7*amin))
    p = 0.3*amax/amin
    # p = 0.2(
    d = 2*amax
    t = 0.4*amax/amin

    return (r, p, d, t)