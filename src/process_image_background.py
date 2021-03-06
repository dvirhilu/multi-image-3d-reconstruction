import cv2
import numpy as np
from scipy.signal import convolve2d
import utils.linalg_utils as linalg
from functools import reduce, cmp_to_key

def apply_harris_corner_filter(image, windowsize=10, sobel_size=3, k=0.04, p=0.1):
    '''
    @brief  Uses a Harris corner detector to generate a boolean valued mask 
            containing 1 clusters around corners in the image

    @param image        Image used for analysis (grayscale)
    @param windowsize   Size of window used for Harris corner detection
    @param sobel_size   Size of sobel kernel used for Harris corner detection
    @param k            The constant used in the Harris corner detector cost 
                        function. Typically between 0.04-0.06
    @param p            The minimum percentage of the maximum value of the 
                        Harris corner detector cost function for which the 
                        pixel is considered a "detected" corner 
    @return             A boolean valued mask containg a clusters of 1 around 
                        coordinates in the image that contain a corner
    '''
    image = np.float32(image)
    dst = cv2.cornerHarris(image,windowsize, sobel_size, k)

    # Threshold for an optimal value, it may vary depending on the image.
    corners = dst > p*dst.max()

    return corners

def squish_cluster_2_centroid(corner_mask):
    '''
    @brief  Find clusters in the boolean corner mask generated by the Harris 
            corner detector and squish them to their centroid

    @param corner_mask  A boolean-valued mask (contains results of Harris 
                        corner detector)
    @return             The updated boolean corner mask
    '''
    rows, cols = corner_mask.shape[:2]

    for i in range(1,rows-1):
        for j in range(1, cols-1):
            if corner_mask[i, j]:
                # get cluster around discovered point
                (y1, y2, x1, x2) = find_cluster(corner_mask, i, i+1, j, j+1)
                # extract window containing cluster
                window = corner_mask[y1:y2, x1:x2]
                # find cluster centroid
                (centroidy, centroidx) = linalg.compute_intensity_centroid(window)

                # create new window with only centroid as 1
                new_window = np.zeros(window.shape, dtype=bool)
                centroidy = int(round(centroidy))
                centroidx = int(round(centroidx))
                new_window[centroidy, centroidx] = True

                # update corner mask cluster to centroid only
                corner_mask[y1:y2, x1:x2] = new_window

    
    return corner_mask

def find_cluster(corner_mask, ystart, ystop, xstart, xstop):
    '''
    @brief  Recursively increments a window around a point in a boolean valued 
            mask until it contains all "touching" points around it

    @param corner_mask  A boolean-valued mask (contains results of Harris 
                        corner detector)
    @param xstart       Should be inputted as x where x is the coordinate of 
                        the initial point for the search
    @param xstop        Should be inputted as x+1 where x is the coordinate of 
                        the initial point for the search
    @param ystart       Should be inputted as y where y is the coordinate of 
                        the initial point for the search
    @param ystop        Should be inputted as y+1 where y is the coordinate of 
                        the initial point for the search
    @return             The window containing the cluster
    '''
    rows, cols = corner_mask.shape[:2]

    # compute incremented bounds in each direction
    ystart_next = ystart-1  if ystart>0         else ystart
    ystop_next  = ystop+1   if ystart<rows-1    else ystop
    xstart_next = xstart-1  if xstart>0         else xstart
    xstop_next  = xstop+1   if xstart<cols-1    else xstop

    # check whether the window contains a nonzero value. If it does, dimension 
    # should be incremented
    top_neighbours = corner_mask[ystart_next, xstart_next:xstop_next]
    inc_top     = reduce(lambda a,b: a or b, top_neighbours)    and ystart > 0    
    
    bottom_neighbours = corner_mask[ystop, xstart_next:xstop_next]
    inc_bottom  = reduce(lambda a,b: a or b, bottom_neighbours) and ystop < rows-1
    
    left_neighbours = corner_mask[ystart_next:ystop_next, xstart_next]
    inc_left    = reduce(lambda a,b: a or b, left_neighbours)   and xstart > 0
    
    right_neighbours = corner_mask[ystart_next:ystop_next, xstop]
    inc_right   = reduce(lambda a,b: a or b, right_neighbours)  and xstop < cols-1    

    # flag that a resize is needed
    increase_window_size =  inc_top or inc_bottom or inc_left or inc_right
    
    if increase_window_size:
        # calcualte new expanded window dimensions based on where expansion is 
        # needed
        ystart  = ystart_next  if inc_top      else ystart
        ystop   = ystop_next   if inc_bottom    else ystop
        xstart  = xstart_next  if inc_left     else xstart
        xstop   = xstop_next   if inc_right    else xstop

        # call function recuresively until window does not need to resize
        return find_cluster(corner_mask, ystart, ystop, xstart, xstop)
    
    # no resize is needed, terminate recursion
    else:
        return (ystart, ystop, xstart, xstop)

def filter_corner_centrosymmetry(image, corner_mask, r, p):
    '''
    @brief  Filters the boolean corner mask to remove points that fail the 
            centrosymmtery criterion of x-corners

    @param image        Image used for analysis (in grayscale)
    @param corner_mask  A boolean-valued mask indicating the location of 
                        detected x-corners
    @param r            The radius of the centrosymmetry filter circular mask 
    @param p            The threshold parameter used in the centrosymmetry 
                        filter
    @return             The filtered boolean corner mask
    '''
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

                # if the window is cutoff cause the corner mask point 
                # is too close to the edge, remove the point
                if corner_neighbourhood.shape != I_masks[0].shape:
                    corner_mask[i,j] = False
                    continue

                I = [
                    np.sum(corner_neighbourhood*mask)
                    for mask in I_masks
                ]

                # average intensity differences between areas in circular mask
                D1 = np.abs(I[0] - I[4]) # D1 = |I1 - I5|
                D2 = np.abs(I[2] - I[6]) # D2 = |I3 - I7|
                D3 = np.abs(I[0] + I[4] - I[2] - I[6])/2 # D3 = |I1 + I5 - I3 - I7|/2
                D4 = np.abs(I[1] - I[5]) # D4 = |I2 - I6|
                D5 = np.abs(I[3] - I[7]) # D5 = |I4 - I8|
                D6 = np.abs(I[1] + I[5] - I[3] - I[7])/2 # D6 = |I2 + I6 - I4 - I8|/2
    
                centrosymmetry_criteria_1_mask = (D1 < p*D3) & (D2 < p*D3)
                centrosymmetry_criteria_2_mask = (D4 < p*D6) & (D5 < p*D6)

                corner_mask[i,j] = centrosymmetry_criteria_1_mask | centrosymmetry_criteria_2_mask

    return corner_mask


def partially_averaging_circular_mask(radius, phi_start, phi_end):
    '''
    @brief  Generate a mask with coefficients to compute the average in a 
            circular slice of specified radius bounded by two specified angles

    @param radius       The radius of the circular slice
    @param phi_start    The starting angle for the circular slice
    @param phi_end      The ending angle for the circular slice
    @return             A radius x radius mask that contains non-zero values 
                        in the averaging region
    '''
    # compares whether the point should be in the averaging region
    def in_averaging_regions(i, j, radius, phi_start, phi_end):
        r, phi = linalg.cart2d_2_pol(i-radius, radius-j) # y coordinate is flipped
        inside_circle  = r <= radius
        
        # account for angular rollover
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

def filter_corner_distance(corner_mask, distance_threshold, num_neighbours=3):
    '''
    @brief  Filters the boolean corner mask to remove points that do not have 
            N neighbours within some threshold distance

    @param corner_mask          A boolean-valued mask indicating the location 
                                of detected x-corners
    @param distance_theshold    The distance threshold used by the filter
    @param num_neighbours       The number of neighbours required to be within 
                                the distance threshold. For x-corners this 
                                should be 3
    @return                     The filtered boolean corner mask
    '''
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
            linalg.get_euclidean_distance(index_pair, neighbour) > distance_threshold
            for neighbour in neighbours
        ]

        above_thresh_exists = reduce(lambda a, b: a or b, distances_above_threshold)

        if above_thresh_exists:
            outmask[index_pair[0], index_pair[1]] = False

    return outmask

def find_k_nearest_neighbours(point, neighbours, k):
    '''
    @brief  Finds a list of kth nearest neighbours based on a euclidean 
            distance criteria

    @param point        The coordinates of the point for which nearest 
                        neighbours should be found 
    @param neighbours   A list of points to search in. This could but is not 
                        required to include "point"
    @param k            The number of nearest neighbours to return
    @return             returns a list of the k-nearest neighbours
    '''
    # copy list to not mutate input
    neighbour_list = neighbours[:]
    
    # sort index list based on distance
    neighbour_list.sort(key=lambda neighbour: linalg.get_euclidean_distance(point, neighbour))

    # check if point is included in the neighbour list
    point_in_neighbour_list = any(
        np.array_equal(point, neighbour) 
        for neighbour in neighbour_list
    )

    # return the k nearest neighbours excluding the point itself
    return neighbour_list[1:k+1] if point_in_neighbour_list else neighbour_list[:k]

def sort_x_corners(corner_mask, image, d):
    '''
    @brief  Sorts x-corners in the list according to a projection invariant 
            order

    @param corner_mask  A boolean-valued mask indicating the location of 
                        detected x-corners   
    @param image        Image used for analysis (in grayscale)
    @param d            Distance threshold for determining whether outer 
                        corners were detected correctly
    @return             Sorted list of x-corners
    '''
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

    sorted_corners_of_interest = sort_outer_corners(image, [A, B, C, D])

    is_succ = True
    for i in range(len(sorted_corners_of_interest)):
        for j in range(len(sorted_corners_of_interest)):
            if i == j:
                continue

            corner1 = sorted_corners_of_interest[i]
            corner2 = sorted_corners_of_interest[j]

            if linalg.get_euclidean_distance(corner1, corner2) < d:
                is_succ = False

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
        sorted_corners_of_interest = sort_outer_corners(image, [A, B, C, D])


    sorted_point_list = []
    for corner in sorted_corners_of_interest:
        neighbours = find_k_nearest_neighbours(corner, corner_list, 3)
        sorted_point_list += sort_corner_neighbourhood(corner, neighbours)

    return sorted_point_list

# TODO: change this to be independent of image center
def sort_corner_neighbourhood(corner_point, corner_neighbourhood):
    '''
    @brief  Sorts points in a cluster in a clockwise order

    @param corner_point outer       x-corner coordinates        
    @param corner_neighbourhood     An unsorted list of the three neighbouring 
                                    x-corners
    @return                         A list containing the sorted corner 
                                    cluster in a clockwise order with the 
                                    outer x-corner as "corner 0"
    '''
    # sort based on negative sin of the angle (clockwise direction)
    def comparator(item1, item2):
        diff1 = item1-corner_point
        diff2 = item2-corner_point

        return linalg.cross_2d(diff1, diff2)

    # sort neighbourhood
    corner_neighbourhood.sort(key=cmp_to_key(comparator))
    # insert outer x-corner to beginning of the list
    corner_neighbourhood.insert(0, corner_point)

    return corner_neighbourhood
    

def sort_outer_corners(image, corners):
    '''
    @brief  Sorts the outer x-corners in a projection invariant order. The 
            corners are sorted in a clockwise order with a consistent choice
            for the outer corner that marks "cluster 0"

    @param image        Image used for analysis (in grayscale)
    @param corners      The computed pixel coordinates of the outer x-corners. 
                        These should already be sorted in clockwise order as
                        [top_left, top_right, bottom_right, bottom_left]
    @return             The ordered outer corner array
    '''
    # decide whether the side pointing towards the top-left corner is the
    # longer side of the board or not
    ab_distance = linalg.get_euclidean_distance(corners[0], corners[1])
    ac_distance = linalg.get_euclidean_distance(corners[1], corners[2])
    long_side_up = ab_distance > ac_distance

    # find centroid of outer-corners
    centroid_point = linalg.compute_point_group_centroid(corners)

    if long_side_up:
        # if long side is pointing to top-left, we are interested to see if
        # top left corner has a black square in the center of its cluster
        corner_of_interest = corners[0]

        # find the colour of the square in the center of the cluster
        movement_direction = linalg.unit_vect(corner_of_interest - centroid_point)
        check_point = (corner_of_interest - 25*movement_direction).astype(int)
        if image[check_point[0], check_point[1]] < 128:
            # central cluster square is black, corner[0] marks "cluster 0"
            return corners
        else:
            # central cluster square is white, corner[2] marks "cluster 0"
            return np.roll(corners, -4)
    else:
        # if long side isn't pointing to top-left, we are interested to see if
        # top right corner has a black square in the center of its cluster
        corner_of_interest = corners[1]

        # find the colour of the square in the center of the cluster
        movement_direction = linalg.unit_vect(corner_of_interest - centroid_point)
        check_point = (corner_of_interest - 25*movement_direction).astype(int)
        if image[check_point[0], check_point[1]] < 128:
            # central cluster square is black, corner[1] marks "cluster 0"
            return np.roll(corners, -2)
        else:
            # central cluster square is white, corner[3] marks "cluster 0"
            return np.roll(corners, -6)

def get_subpixel_coordinates(image, point_list, windowsize=10):
    '''
    @brief  Retrieves the sub-pixel coordinates of each x-corner point using
            the squared grayscale intensity centroid method

    @param image        Image used for analysis (in grayscale)
    @param windowsize   Size of window for which the centroid is computed
    @param point_list   The computed pixel coordinates of the x-corners 
    @return             The sub-pixel coordinates of x-corners
    '''
    subpixel_points = []
    for point in point_list:
        i = point[0]
        j = point[1]
        r = int(windowsize/2)
        window = image[i-r:i+r, j-r:j+r]
        (cy, cx) = linalg.compute_intensity_centroid(window**2)

        # centroid within window, need to shift to image point
        cx += j-r
        cy += i-r

        subpixel_points.append(np.array([cx, cy]))

    return subpixel_points


def get_ordered_image_points(image, windowsize=10 ,sobel_size=3, k=0.04, harris_threshold=0.1, r=40, p=0.5, d=150):
    '''
    @brief  Locates x-corners used for extrinsic parameter camera calibration 
            and returns them in a projection-invariant order

    @param image            Image used for analysis
    @param windowsize       Size of window used for Harris corner detection
    @param sobel_size       Size of sobel kernel used for Harris corner 
                            detection
    @param k                The constant used in the Harris corner detector
                            cost function. Typically between 0.04-0.06
    @param harris_threshold The minimum percentage of the maximum value of the 
                            Harris corner detector cost function for which the 
                            pixel is considered a "detected" corner 
    @param r                The radius of the centrosymmetry filter circular 
                            mask 
    @param p                The threshold parameter used in the centrosymmetry 
                            filter 
    @param d                The distance threshold used to filter false x-corners
    
    @return sorted_point_list   A sorted list of the x-corner sub-pixel image 
                                coordinates
    @return is_valid            Indicates whether sufficient featuers could be
                                found in the image to generate the point list
    '''

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
    sorted_point_list = sort_x_corners(corner_mask, image, d)

    # get subpixel location of points
    sorted_point_list = get_subpixel_coordinates(image, sorted_point_list, windowsize=windowsize)

    # TODO: create function to automatically check validity of results

    return (True, sorted_point_list)

def get_undistored_k_matrix(image, k, d):
    '''
    @brief  Retrieves an adjusted instrisic camera calibration parameter 
            matrix after removing image distortion. 
    
    For more info:
    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    @param image    The image to undistort
    @param k        The unadjusted 3x3 intrinsic camera calibration matrix
    @param d        The distortion parameters, in a (5,) shaped ndarray in the
                    format [k1, k2, p1, p2, p3]
    @return         The adjusted 3x3 intrinsic camera calibration matrix
    '''
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w,h), 1, (w,h))

    return newcameramtx, roi

def undistort(image, k, d, k_adj, roi):
    '''
    @brief  Performs the undistorting operation on the image 
    
    For more info:
    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    @param image    The image to undistort
    @param k        The unadjusted 3x3 intrinsic camera calibration matrix
    @param d        The distortion parameters, in a (5,) shaped ndarray in the
                    format [k1, k2, p1, p2, p3]
    @param k_adj    The adjusted 3x3 intrinsic camera calibration matrix
    @param roi      The region of interest produced by K matrix adjustment
    @return         The undistorted image
    '''
    # undistort
    undst = cv2.undistort(image, k, d, None, k_adj)
    # crop the image
    x, y, w, h = roi
    undst = undst[y:y+h, x:x+w]

    return undst