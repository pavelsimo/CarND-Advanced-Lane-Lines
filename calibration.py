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


def plot_calibration(filename):
    img = cv2.imread(filename)
    objpoints, imgpoints = load_calibration_parameters(6, 9)
    img_undistort = undistort(img, objpoints, imgpoints)
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=50)

    ax2.imshow(cv2.cvtColor(img_undistort, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image', fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


if __name__ == '__main__':
    plot_calibration('test_images/test3.jpg')
