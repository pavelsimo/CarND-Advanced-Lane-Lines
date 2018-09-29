from moviepy.editor import VideoFileClip
from cv import *
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Lane(object):
    def __init__(self):
        self.samples = 10
        self.left_fits = [0] * self.samples
        self.right_fits = [0] * self.samples
        self.left_fit = None
        self.right_fit = None
        self.fit_count = 0
        self.fit_index = 0
        self.skip = False
        objpoints, imgpoints = load_calibration_parameters(6, 9)
        self.objpoints = objpoints
        self.imgpoints = imgpoints
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

    def fit_polynomial(self, img, verbose=0):
        # find our lane pixels first
        leftx, lefty, rightx, righty, dist_center_px, out_img = self.find_lane_pixels(img)

        # fitting a second order polynomial
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.skip = False
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except Exception as e:
            self.skip = True
            logger.warning('skipping this frame...')
            logger.warning(e)

        y_eval = np.max(ploty)
        left_curverad, right_curverad = self.measure_curvature_real(y_eval, leftx, lefty, rightx, righty)

        outlier_threshold = 50
        if self.fit_count > self.samples and (left_curverad < outlier_threshold or right_curverad < outlier_threshold):
            self.skip = True

        if self.skip:
            pass  # do nothing...
        else:
            index = self.fit_count % self.samples
            self.left_fits[index] = left_fit
            self.right_fits[index] = right_fit

            if self.fit_count >= self.samples:
                self.left_fit = np.mean(np.array(self.left_fits), axis=0)
                self.right_fit = np.mean(np.array(self.right_fits), axis=0)
            else:
                self.left_fit = left_fit
                self.right_fit = right_fit

            self.fit_count += 1

        try:
            left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        # colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        if verbose:
            plt.plot(left_fitx, ploty, color='green')
            plt.plot(right_fitx, ploty, color='green')
            plt.gca().invert_yaxis()  # to visualize as we do the images

        return left_fitx, right_fitx, left_curverad, right_curverad, dist_center_px, out_img

    def find_lane_pixels(self, img, nwindows=9, margin=100, minpix=50):

        # take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

        # create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # calculating the lane distance from the center of the image
        height, width = img.shape[0], img.shape[1]
        lane_center = leftx_base + (rightx_base - leftx_base) // 2
        # positive: right of the center of the image
        # negative: right of the center of the image
        img_center = (width // 2)
        dist_center_px = lane_center - img_center

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

        return leftx, lefty, rightx, righty, dist_center_px, out_img

    def measure_curvature_real(self, y_eval, leftx, lefty, rightx, righty):
        left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(
            2 * right_fit_cr[0])
        return left_curverad, right_curverad

    def unwrap_lane(self, undist, warped, left_fitx, right_fitx, Minv, verbose=0):
        # create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

        # combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        if verbose:
            plt.imshow(newwarp)
            plt.show()
            plt.imshow(result)
            plt.show()

        return result

    def draw(self, img):
        img_undistort = undistort(img, self.objpoints, self.imgpoints)
        result = np.copy(img_undistort)

        # wrap area
        height, width = img.shape[0], img.shape[1]
        src = [
            ((width // 2) - 70, (height // 2) + 110),
            ((width // 2) + 80, (height // 2) + 110),
            (width - 170, height),
            (0 + 220, height)
        ]
        dst = [
            (width // 4, 0),
            (width - width // 4 - 40, 0),
            (width - width // 4 - 40, height),
            (width // 4, height)
        ]

        # step 2: gradient and threshold
        result = compose_threshold(result)
        img_sobel = np.copy(result)

        # step 3: select region of interest
        vertices = np.int32([[
            (src[0][0] - 60, src[0][1]),
            (src[1][0] + 60, src[1][1]),
            (src[2][0] + 60, src[2][1] - 50),
            (src[3][0] - 60, src[3][1] - 50)]]
        )
        result = region_of_interest(result, vertices)

        # step 4: perspective transform
        result, M, Minv = unwarp(result, np.float32(src), np.float32(dst))

        # step 5: fit poly
        left_fit, right_fit, left_curverad, right_curverad, dist_center_px, img_poly = self.fit_polynomial(result)

        # step 6: draw lane
        result = self.unwrap_lane(img_undistort, result, left_fit, right_fit, Minv)

        # step 7: draw pipeline visualizations
        img_overlay1 = cv2.resize(img_poly, (256, 144))
        img_overlay2 = cv2.resize(cv2.cvtColor(255 * img_sobel, cv2.COLOR_GRAY2RGB), (256, 144))
        result = overlay(result, img_overlay1, x_offset=50, y_offset=80)
        result = overlay(result, img_overlay2, x_offset=316, y_offset=80)
        text = 'LCurvature: %.3fm RCurvature: %.3fm Center Dist: %.3fcm' % (left_curverad, right_curverad,
                                                                            dist_center_px * self.xm_per_pix * 100)
        result = put_text(result, text, x_offset=50, y_offset=50)

        return result


if __name__ == '__main__':

    line = Lane()

    def pipeline(img):
        try:
            result = line.draw(img)
        except Exception as e:
            logger.exception(e)
            result = img
        return result

    clip1 = VideoFileClip("project_video.mp4").subclip(0, 1)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile("output_videos/project_video2.mp4", audio=False)
