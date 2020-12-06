import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import utils.plt_utils as plt_utils
from utils.linalg_utils import get_euclidean_distance, get_vec_cos_angle, cart2d_2_pol, unit_vect, cross_2d
from functools import reduce, cmp_to_key
import utils.file_io_utils as file_io_utils

def apply_harris_corner_filter(image, windowsize=10, sobel_size=3, k=0.04, use_cv=True, p=0.1):
    if use_cv:
        image = np.float32(image)
        dst = cv2.cornerHarris(image,windowsize, sobel_size, k)

        # Threshold for an optimal value, it may vary depending on the image.
        corners = dst > p*dst.max()

        return corners

def squish_cluster_2_centroid(corner_mask):
    rows, cols = corner_mask.shape[:2]

    for i in range(1,rows-1):
        for j in range(1, cols-1):
            if corner_mask[i, j]:
                # get cluster around discovered point
                (y1, y2, x1, x2) = find_cluster(corner_mask, i, i+1, j, j+1)
                # extract window containing cluster
                window = corner_mask[y1:y2, x1:x2]
                # find cluster centroid
                (centroidy, centroidx) = find_centroid(window)

                # create new window with only centroid as 1
                new_window = np.zeros(window.shape, dtype=bool)
                centroidy = int(round(centroidy))
                centroidx = int(round(centroidx))
                new_window[centroidy, centroidx] = True

                # update corner mask cluster to centroid only
                corner_mask[y1:y2, x1:x2] = new_window

    
    return corner_mask

def find_centroid(window):
    rows, cols = window.shape[:2]

    total_mass = np.sum(window)
    y_centroid = 0
    x_centroid = 0
    for i in range(rows):
        for j in range(cols):
            mass = window[i,j]
            y_centroid += i*mass / total_mass
            x_centroid += j*mass / total_mass
    
    return(y_centroid, x_centroid)


def find_cluster(corner_mask, ystart, ystop, xstart, xstop):
    rows, cols = corner_mask.shape[:2]

    ystart_next = ystart-1  if ystart>0         else ystart
    ystop_next  = ystop+1   if ystart<rows-1    else ystop
    xstart_next = xstart-1  if xstart>0         else xstart
    xstop_next  = xstop+1   if xstart<cols-1    else xstop

    top_neighbours = corner_mask[ystart_next, xstart_next:xstop_next]
    bottom_neighbours = corner_mask[ystop, xstart_next:xstop_next]
    left_neighbours = corner_mask[ystart_next:ystop_next, xstart_next]
    right_neighbours = corner_mask[ystart_next:ystop_next, xstop]

    inc_top     = reduce(lambda a,b: a or b, top_neighbours)    and ystart > 0    
    inc_bottom  = reduce(lambda a,b: a or b, bottom_neighbours) and ystop < rows-1
    inc_left    = reduce(lambda a,b: a or b, left_neighbours)   and xstart > 0
    inc_right   = reduce(lambda a,b: a or b, right_neighbours)  and xstop < cols-1    

    # flag that a resize is needed
    increase_window_size =  inc_top or inc_bottom or inc_left or inc_right
    
    if increase_window_size:
        # calcualte new expanded window dimensions based on where expansion is needed
        ystart  = ystart_next  if inc_top      else ystart
        ystop   = ystop_next   if inc_bottom    else ystop
        xstart  = xstart_next  if inc_left     else xstart
        xstop   = xstop_next   if inc_right    else xstop

        # call function recuresively until window does not need to resize
        return find_cluster(corner_mask, ystart, ystop, xstart, xstop)
    
    else:
        return (ystart, ystop, xstart, xstop)

def filter_corner_centrosymmetry(image, corner_mask, r, p):
    I_masks = [
        partially_averaging_circular_mask(r, i*360/8, (i+1)*360/8)
        for i in range(8)
    ]

    rows, cols = corner_mask.shape[:2]

    for i in range(rows):
        for j in range(cols):
            if corner_mask[i,j]:
                if i < r or j < r:
                    corner_mask[i, j] = False
                    continue

                corner_neighbourhood = image[i-r:i+r, j-r:j+r]

                I = [
                    np.sum(corner_neighbourhood*mask)
                    for mask in I_masks
                ]

                # print(I)

                # average intensity differences between areas in circular mask
                D1 = np.abs(I[0] - I[4]) # D1 = |I1 - I5|
                D2 = np.abs(I[2] - I[6]) # D2 = |I3 - I7|
                D3 = np.abs(I[0] + I[4] - I[2] - I[6])/2 # D3 = |I1 + I5 - I3 - I7|/2
                D4 = np.abs(I[1] - I[5]) # D4 = |I2 - I6|
                D5 = np.abs(I[3] - I[7]) # D5 = |I4 - I8|
                D6 = np.abs(I[1] + I[5] - I[3] - I[7])/2 # D6 = |I2 + I6 - I4 - I8|/2

                # print(D1, D2, D3, D4, D5, D6)
    
                centrosymmetry_criteria_1_mask = (D1 < p*D3) & (D2 < p*D3)
                centrosymmetry_criteria_2_mask = (D4 < p*D6) & (D5 < p*D6)

                corner_mask[i,j] = centrosymmetry_criteria_1_mask | centrosymmetry_criteria_2_mask

    # output how many corners were detected
    # print(np.sum(corner_mask))

    return corner_mask


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
    x = np.where(corner_mask)[1]
    y = np.where(corner_mask)[0]

    corner_indices = [
        np.array([y_val, x_val])
        for (y_val, x_val) in zip(y,x)
    ]

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
        get_euclidean_distance(index_pair, neighbour)
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

    # print(mu, sigma, amin, amax)

    # calcualte threshold values
    r = int(round(0.7*amin))
    p = 0.3*amax/amin
    # p = 0.2(
    d = 2*amax
    t = 0.4*amax/amin

    return (r, p, d, t)

def get_desired_chessboard_points(corner_mask, image, d):
    x = np.where(corner_mask)[1]
    y = np.where(corner_mask)[0]

    corner_list = [
        np.array([y_val, x_val])
        for (y_val, x_val) in zip(y,x)
    ]

    # case 2 - image is aligned so points are extrema of x-y and x+y
    # try this approach and see if it fails
    sum_arr = x + y
    diff_arr = y - x

    A_index = np.argmin(sum_arr)
    B_index = np.argmin(diff_arr)
    C_index = np.argmax(sum_arr)
    D_index = np.argmax(diff_arr)

    A = np.array([y[A_index], x[A_index]])
    B = np.array([y[B_index], x[B_index]])
    C = np.array([y[C_index], x[C_index]])
    D = np.array([y[D_index], x[D_index]])

    sorted_corners_of_interest = sort_corners(image, [A, B, C, D])

    is_succ = True
    for i in range(len(sorted_corners_of_interest)):
        for j in range(len(sorted_corners_of_interest)):
            if i == j:
                continue

            corner1 = sorted_corners_of_interest[i]
            corner2 = sorted_corners_of_interest[j]

            if get_euclidean_distance(corner1, corner2) < d:
                is_succ = False

    # print("is successful: ", is_succ)

    if not is_succ:
        # case 2 - image is tilted so points are extrema of x and y

        A_index = np.argmin(x)
        B_index = np.argmin(y)
        C_index = np.argmax(x)
        D_index = np.argmax(y)

        A = np.array([y[A_index], x[A_index]])
        B = np.array([y[B_index], x[B_index]])
        C = np.array([y[C_index], x[C_index]])
        D = np.array([y[D_index], x[D_index]])

        # check whether this is correct
        sorted_corners_of_interest = sort_corners(image, [A, B, C, D])


    sorted_point_list = []
    for corner in sorted_corners_of_interest:
        neighbours = find_k_nearest_neighbours(corner, corner_list, 3)
        sorted_point_list += sort_corner_neighbourhood(corner, neighbours)

    outmask = np.zeros(corner_mask.shape, dtype=bool)
    for corner in sorted_point_list:
        outmask[corner[0], corner[1]] = True

    return (sorted_point_list, outmask)

# TODO: change this to be independent of image center
def sort_corner_neighbourhood(corner_point, corner_neighbourhood):
    # sort based on negative sin of the angle (clockwise direction)
    def comparator(item1, item2):
        diff1 = item1-corner_point
        diff2 = item2-corner_point

        return cross_2d(diff1, diff2)

    corner_neighbourhood.sort(key=cmp_to_key(comparator))
    corner_neighbourhood.insert(0, corner_point)

    return corner_neighbourhood
    

def sort_corners(image, corners):
    ab_distance = get_euclidean_distance(corners[0], corners[1])
    ac_distance = get_euclidean_distance(corners[1], corners[2])
    # print(ab_distance, ac_distance)
    long_side_up = ab_distance > ac_distance

    rows, cols = image.shape[:2]

    if long_side_up:
        corner_of_interest = corners[0]
        center_point = np.array([int(rows/2), int(cols/2)])

        movement_direction = unit_vect(corner_of_interest - center_point)
        
        check_point = (corner_of_interest - 25*movement_direction).astype(int)
        # print(check_point)
        # print(image[check_point[0], check_point[1]])

        if image[check_point[0], check_point[1]] < 128:
            # in this case, first corner is "corner 1"
            return corners
        else:
            # in this case, the opposite corner is "corner 1"
            return np.roll(corners, -4)
    else:
        corner_of_interest = corners[1]
        center_point = np.array([int(rows/2), int(cols/2)])

        movement_direction = unit_vect(corner_of_interest - center_point)
        
        check_point = (corner_of_interest - 25*movement_direction).astype(int)
        # print(check_point)
        # print(image[check_point[0], check_point[1]])

        if image[check_point[0], check_point[1]] < 128:
            return np.roll(corners, -2)
        else:
            return np.roll(corners, -6)

def get_subpixel_coordinates(image, point_list, windowsize=10):
    subpixel_points = []
    for point in point_list:
        i = point[0]
        j = point[1]
        r = int(windowsize/2)
        window = image[i-r:i+r, j-r:j+r]
        (cy, cx) = find_centroid(window)

        # centroid within window, need to shift to image point
        cx += j-r
        cy += i-r

        subpixel_points.append(np.array([cx, cy]))

    return subpixel_points


def get_ordered_image_points(image, windowsize=10 ,sobel_size=3, k=0.04, harris_threshold=0.1, r=40, p=0.5, d=150):

    # use Harris corner detection to get initial corner distribution
    corner_mask = apply_harris_corner_filter(image, windowsize=windowsize, sobel_size=sobel_size, k=k, p=harris_threshold)

    # Harris corner will generate clusters of points. Find clusters and squish into the centroid of the cluster
    corner_mask = squish_cluster_2_centroid(corner_mask)

    # to eliminate fake corners outside of chessboard pattern, filter for centrosymmetry property
    # this will still leave some fake corners on object in the middle, but the middle of the chessboard will be ignored
    corner_mask = filter_corner_centrosymmetry(image, corner_mask, r, p)

    # for presistent fake corners, use a distance filter
    filter_corner_distance(corner_mask, d, 3)

    # if all corners have been filtered out, indicate that corner searching failed in return value
    if np.count_nonzero(corner_mask)==0:
        return (False, None)

    # since the object points are hard to filter out, only gather corners of the chessboard and 3 nearest neighbours
    sorted_point_list, corner_mask = get_desired_chessboard_points(corner_mask, image, d)

    # get subpixel location of points
    sorted_point_list = get_subpixel_coordinates(image, sorted_point_list, windowsize=windowsize)

    # TODO: create function to automatically check validity of results

    return (True, sorted_point_list)

def get_undistored_k_matrix(image, k, d):
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 1, (w,h))

    return newcameramtx, roi

def undistort(image, k, d, k_adj, roi):
    # undistort
    undst = cv2.undistort(image, k, d, None, k_adj)
    # crop the image
    x, y, w, h = roi
    undst = undst[y:y+h, x:x+w]

    return undst

if __name__=="__main__":

    images = file_io_utils.load_object_images("eraser")
    # good_indices = [0, 2, 4, 5, 7, 10, 11]
    # good_indices = [0, 2]
    # images = [
    #     images[i] 
    #     for i in good_indices
    # ]
    images = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for image in images
    ]

    titles = [
        "image %0d" % i
        for i in range(len(images))
    ]

    plt_utils.show_images(*images, titles=titles, sup_title="Original Image Set")
    windowsize=10
    sobel_size=3
    k=0.04
    harris_threshold=0.1
    r=40
    p=0.5
    d = 150

    corners = [
        apply_harris_corner_filter(image, windowsize=windowsize, sobel_size=sobel_size, k=k, p=harris_threshold)
        for image in images[11:13]
    ]

    plt_utils.plot_corner_points(images[11:13], corners, titles=titles[11:13], sup_title="Harris Corners")

    corners = [
        squish_cluster_2_centroid(corner_mask)
        for corner_mask in corners
    ]

    plt_utils.plot_corner_points(images[11:13], corners, titles=titles[11:13], sup_title="Squished Harris Corners")

    corners = [
        filter_corner_centrosymmetry(image, corner_mask, r, p)
        for (corner_mask, image) in zip(corners, images[11:13])
    ]

    plt_utils.plot_corner_points(images[11:13], corners, titles=titles[11:13], sup_title="Centrosymmetry Filter")

    corners = [
        filter_corner_distance(corner_mask, d, 3)
        for corner_mask in corners
    ]

    plt_utils.plot_corner_points(images[11:13], corners, titles=titles[11:13], sup_title="Distance Filter")
    

    ret_tuples = [
        get_ordered_image_points(image, windowsize=windowsize, sobel_size=sobel_size, k=k, harris_threshold=harris_threshold, r=r, p=p, d=d)
        for image in images
    ]

    is_valids = [
        ret_tuple[0]
        for ret_tuple in ret_tuples
    ]
    print(is_valids)

    image_points = [
        ret_tuple[1]
        for (ret_tuple, is_valid) in zip(ret_tuples, is_valids)
        if is_valid
    ]

    corners = [
        ret_tuple[2]
        for (ret_tuple, is_valid) in zip(ret_tuples, is_valids)
        if is_valid
    ]
    
    images = [
        image
        for (image, is_valid) in zip(images, is_valids)
        if is_valid
    ]

    titles = [
        title
        for (title, is_valid) in zip(titles, is_valids)
        if is_valid
    ]


    print(len(corners), len(image_points), np.shape(ret_tuples), np.shape(is_valids))

    # show desired corners
    plt_utils.plot_point_path(images[11:13], corners[11:13], image_points[11:13], titles=titles, sup_title="Corner Point Sequence")
    plt_utils.plot_image_points(images[3:5], image_points[3:5], titles=titles[3:5], sup_title="Final Corner Points", same_colour=False)
    
    plt.show()