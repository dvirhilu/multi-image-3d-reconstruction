import cv2
import numpy as np
from imutils import rotate_bound
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib import cm
from utils.im_view_utils import show_images

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

    show_images(logmag(H), logmag(imfft), logmag(imfft_filtered))
    
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

def apply_hessian_det_mask(image):
    fxx = fxx_mask()
    fyy = fyy_mask()
    fxy = fxy_mask()

    image_xx = convolve2d(image, fxx, mode='same')
    image_yy = convolve2d(image, fyy, mode='same')
    image_xy = convolve2d(image, fxy, mode='same')

    return image_xx*image_xy - image_xy**2

def add_noise(im, sigma):
    rows, cols = im.shape

    for i in range(rows):
        for j in range(cols):
            x = i - rows/2
            y = j - cols/2
            r = np.sqrt(x**2 + y**2)
            exponent = -0.5*(r/sigma)**2
            # normalizer = 1/(2*np.pi*sigma**2)
            normalizer = 1
            should_swap = np.random.uniform() < normalizer*np.e**exponent

            if should_swap:
                im[i][j] = np.random.randint(0,256)
    
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
    im = apply_gaussian_filter(im, 0.05)
    im_rot = apply_gaussian_filter(im_rot, 0.05)
    im_aff = apply_gaussian_filter(im_aff, 0.05)

    im_titles = [
        title + " Blurred"
        for title in im_titles_orig
    ]
    
    # show blurred images
    show_images(im, im_rot, im_aff, titles=im_titles)

    # apply hessian determinate mask
    im = apply_hessian_det_mask(im)
    im_rot = apply_hessian_det_mask(im_rot)
    im_aff = apply_hessian_det_mask(im_aff)

    im_titles = [
        title + " HessDet"
        for title in im_titles_orig
    ]

    # show hessian'd images
    show_images(im, im_rot, im_aff, titles=im_titles)