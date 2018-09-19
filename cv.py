import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Line2D(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @staticmethod
    def from_lst(lst):
        return Line2D(lst[0], lst[1], lst[2], lst[3])

    @staticmethod
    def from_vec2d(v1, v2):
        return Line2D(v1.x, v1.y, v2.x, v2.y)

    def to_lst(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def __str__(self):
        return '(%.2f, %.2f, %.2f, %.2f)' % (self.x1, self.y1, self.x2, self.y2)


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


def draw_line(img, line, color=None, thickness=2):
    color = [255, 0, 0] if color is None else color
    draw_lines(img, [[[int(x) for x in line.to_lst()]]], color=color, thickness=thickness)


def draw_point(img, p, color=None):
    color = [255, 0, 0] if color is None else color
    cv2.circle(img, (p[0], p[1]), 30, color, -1)


def draw_region_of_interest(img):
    vertices = np.array([[(0, img.shape[0]), (450, 290), (490, 290),
                          (img.shape[1], img.shape[0])]], dtype=np.int32)


def unwarp(undis, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undis, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def plot_comparison(img1, img2, title1='', title2='', cmap1=None, cmap2=None):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=50)

    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


if __name__ == '__main__':

    #
    # step1: undistort image
    #

    img = cv2.imread('test_images/test3.jpg')
    objpoints, imgpoints = load_calibration_parameters(6, 9)
    img_undistort = undistort(img, objpoints, imgpoints)
    plot_comparison(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_undistort, cv2.COLOR_BGR2RGB),
        title1='Original Image',
        title2='Undistorted Image'
    )

    #
    # step 2: select region of interest
    #

    vertices = np.array([[(0, img.shape[0] - 50), (550, 430), (720, 430),
                          (img.shape[1], img.shape[0] - 50)]], dtype=np.int32)

    img_crop = region_of_interest(img_undistort, vertices)
    plot_comparison(
        cv2.cvtColor(img_undistort, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB),
        title1='Undistorted Image',
        title2='Crop Image',
        cmap2='gray'
    )

    #
    # step 3: gradient and threshold
    #

    img_sobel = compose_threshold(img_crop, s_thresh=(170, 255), thresh=(50, 255))
    plot_comparison(
        cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB),
        img_sobel,
        title1='Crop Image',
        title2='Gradient Image',
        cmap2='gray'
    )

    #
    # step 4: perspective transform
    #

    src = np.array([(0, img.shape[0] - 50), (300, 550), (950, 550), (img.shape[1], img.shape[0] - 50)], dtype=np.float32)
    dst = np.array([(0, img.shape[0]), (0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0])], dtype=np.float32)
    img_unwarp = unwarp(img_sobel, src, dst)
    plot_comparison(
        img_sobel,
        img_unwarp,
        title1='Gradient Image',
        title2='Unwrap Image',
        cmap1='gray',
        cmap2='gray'
    )
