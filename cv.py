import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


def undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
    return img_undistort


def load_calibration_parameters(height, width, verbose=0):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # if found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if verbose == 1:
                # draw and display the corners
                img = cv2.drawChessboardCorners(img, (width, height), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

    return objpoints, imgpoints


def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(np.pi / 3, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def compose_threshold(img, s_thresh=(170, 255), thresh=(20, 100)):
    # convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # take the derivative in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    # absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.abs(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def hue_threshold(img, thresh=(90, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(h)
    binary_output[(h > thresh[0]) & (h <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def combine_threshold(binary1, binary2):
    """ combine the two binary thresholds"""
    combined_binary = np.zeros_like(binary1)
    combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
    return combined_binary

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=None, thickness=2):
    color = [255, 0, 0] if color is None else color
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_point(img, p, color=None):
    color = [255, 0, 0] if color is None else color
    cv2.circle(img, (p[0], p[1]), 30, color, -1)


def unwarp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def histogram(img):
    h, w = img.shape[0], img.shape[1]
    offset = 200
    bottom_half = img[h // 2:h, :]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


def plot_comparison(img1, img2, title1='', title2='', cmap1=None, cmap2=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=50)

    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def plot_histogram(y):
    plt.plot(y)
    plt.show()


def find_lane_pixels(img, nwindows=9, margin=100, minpix=50):
    """

    :param img:
    :param nwindows: number of sliding windows
    :param margin: width of the windows +/- margin
    :param minpix: minimum number of pixels found to recenter window
    :return:
    """

    # take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # find the peak of the left and right halves of the histogram
    # these will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0] // nwindows)

    # identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # step through the windows one by one
    for window in range(nwindows):
        # identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height

        # find the four below boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if you found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(img, verbose=0):
    # find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)

    # fitting a second order polynomial
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    y_eval = np.max(ploty)
    left_curverad, right_curverad = measure_curvature_real(y_eval, leftx, lefty, rightx, righty)
    print(left_curverad, right_curverad)

    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='green')
    plt.gca().invert_yaxis()  # to visualize as we do the images

    return left_fitx, right_fitx, out_img


def measure_curvature_real(y_eval, leftx, lefty, rightx, righty, ym_per_pix=30/720, xm_per_pix =3.7/700):
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(
        2 * right_fit_cr[0])
    return left_curverad, right_curverad


def draw_lane(undist, warped, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    plt.imshow(newwarp)
    plt.show()
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':

    #
    # step1: undistort image
    #
    #img = cv2.imread('test_images/straight_lines1.jpg')
    img = cv2.imread('test_images/test4.jpg')
    objpoints, imgpoints = load_calibration_parameters(6, 9)
    img_undistort = undistort(img, objpoints, imgpoints)

    # DEBUG
    height, weight = img.shape[0], img.shape[1]
    # src = [(600, 445),  (680, 445), (1135, height), (185, height)]
    # src = [(600, 445),  (680, 445), (weight - 100, height), (100, height)]
    # src = [(570, 470), (722, 470), (1110, 720), (220, 720)]
    src = [(570, 470), (722, 470), (1110, 720), (220, 720)]

    #dst = [(310, 0), (980, 0), (980, height), (310, height)]
    dst = [(320, 0), (920, 0), (920, 720), (320, 720)]
    #              blue         green        red         cyan
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    for point, color in zip(src, colors):
        draw_point(img, (point[0], point[1]), color=color)


    plot_comparison(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_undistort, cv2.COLOR_BGR2RGB),
        title1='Original Image',
        title2='Undistorted Image'
    )

    #
    # step 2: gradient and threshold
    #

    img_sobel1 = compose_threshold(img_undistort, s_thresh=(170, 255), thresh=(50, 100))
    img_sobel2 = hue_threshold(img_undistort, thresh=(11, 39))  # yellow
    img_sobel = combine_threshold(img_sobel1, img_sobel2)
    plot_comparison(
        cv2.cvtColor(img_undistort, cv2.COLOR_BGR2RGB),
        img_sobel,
        title1='Crop Image',
        title2='Gradient Image',
        cmap2='gray'
    )

    #
    # step 3: select region of interest
    #


    #vertices = np.array([[(690, 440), (1136, 720), (177, 720), (580, 440)]],dtype=np.int32)
    #vertices = np.array([[(0, img.shape[0] - 50), (550, 450), (720, 450), (img.shape[1], img.shape[0] - 50)]], dtype=np.int32)

    src = [(570, 470), (722, 470), (1110, 720), (220, 720)]
    vertices = np.int32([[
        (src[0][0] - 60, src[0][1]),
        (src[1][0] + 60, src[1][1]),
        (src[2][0] + 60, src[2][1]),
        (src[3][0] - 60, src[3][1])]]
    )
    img_crop = region_of_interest(img_sobel, vertices)
    plot_comparison(
        img_sobel,
        img_crop,
        title1='Undistorted Image',
        title2='Crop Image',
        cmap1='gray',
        cmap2='gray'
    )

    #
    # step 4: perspective transform
    #

    #src = np.array([(0, img.shape[0] - 50), (300, 470), (950, 470), (img.shape[1], img.shape[0] - 50)], dtype=np.float32)
    img_unwarp, M, Minv = unwarp(img_crop, np.float32(src), np.float32(dst))
    plot_comparison(
        img_crop,
        img_unwarp,
        title1='Gradient Image',
        title2='Unwrap Image',
        cmap1='gray',
        cmap2='gray'
    )

    #
    # step 5: image histogram
    #
    h = histogram(img_unwarp)
    plot_histogram(h)

    #
    # step 6: fit poly
    #
    left_fit, right_fit, img_poly = fit_polynomial(img_unwarp, verbose=1)
    plot_comparison(
        img_unwarp,
        img_poly,
        title1='Unwrap Image',
        title2='Poly Image',
        cmap1='gray',
        cmap2='gray'
    )

    #
    # step 8: draw lane
    #
    draw_lane(img_undistort, img_unwarp, left_fit, right_fit, Minv)

