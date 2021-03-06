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


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # take the gradient in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # calculate the magnitude
    mag_sobel = np.sqrt(sobelx ** 2)

    # scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))

    # create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # inversed the mask, so can be combined with color threshold later on...
    binary_output = 1 - binary_output
    return binary_output


def overlay(img1, img2, x_offset=50, y_offset=50):
    img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
    return img1


def put_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, x_offset=0, y_offset=0, scale=1, color=(255, 255, 255), line_type=2):
    result = np.copy(img)
    return cv2.putText(result, text, (x_offset, y_offset), font, scale, color, line_type)


def white_threshold(img, thresh=(0, 255)):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    binary_output = np.zeros_like(r)
    binary_output[(r >= thresh[0]) & (r <= thresh[1]) &
                  (g >= thresh[0]) & (g <= thresh[1]) &
                  (b >= thresh[0]) & (b <= thresh[1])] = 1
    return binary_output


def channel_threshold(img, channel=0, thresh=(0, 255)):
    c = img[:, :, channel]
    binary_output = np.zeros_like(c)
    binary_output[(c >= thresh[0]) & (c <= thresh[1])] = 1
    return binary_output


def combine_threshold(binary1, binary2):
    combined_binary = np.zeros_like(binary1)
    combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
    return combined_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # take the gradient in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # calculate the magnitude
    mag_sobel = np.sqrt(sobelx ** 2)

    # scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))

    # create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # inversed the mask, so can be combined with color threshold later on...
    binary_output = 1 - binary_output
    return binary_output


def compose_threshold(img, verbose=0):
    # white threshold
    img1 = white_threshold(img, thresh=(190, 255))
    # red threshold
    img2 = channel_threshold(img, channel=0, thresh=(220, 255))
    # sobel-x
    img3 = mag_thresh(img, sobel_kernel=21, thresh=(0, 40))
    # yellow threshold
    img4 = channel_threshold(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), channel=0, thresh=(20, 30))

    result = combine_threshold(img1, img2)
    result = combine_threshold(result, img3)
    result = combine_threshold(result, img4)

    if verbose:
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(40, 30))
        ax1.set_title('white threshold')
        ax1.imshow(img1, cmap='gray')
        ax2.set_title('red threshold')
        ax2.imshow(img2, cmap='gray')
        ax3.set_title('sobel-x')
        ax3.imshow(img3, cmap='gray')
        ax4.set_title('yellow threshold')
        ax4.imshow(img3, cmap='gray')
        ax5.set_title('binary result')
        ax5.imshow(result, cmap='gray')

    return result


def region_of_interest(img, vertices):
    """
    Crop the region of interest, only keeps the region of the image defined by the polygon
    formed from `vertices`.
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


def warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def histogram(img):
    h, w = img.shape[0], img.shape[1]
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


def plot_image_rgb_channels(rgb):
    # Isolate RGB channels
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Visualize the individual color channels
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.set_title('R channel')
    ax1.imshow(r, cmap='gray')
    ax2.set_title('G channel')
    ax2.imshow(g, cmap='gray')
    ax3.set_title('B channel')
    ax3.imshow(b, cmap='gray')

