from moviepy.editor import VideoFileClip
from cv import *
import numpy as np
import logging

logger = logging.getLogger(__name__)


objpoints, imgpoints = load_calibration_parameters(6, 9)


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

    def fit_polynomial(self, img, verbose=0):
        # find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)

        # fitting a second order polynomial
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.skip = False
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except Exception as e:
            skip = True
            logger.warning('skipping this frame...')
            logger.warning(e)

        y_eval = np.max(ploty)
        left_curverad, right_curverad = measure_curvature_real(y_eval, leftx, lefty, rightx, righty)

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

        return left_fitx, right_fitx, left_curverad, right_curverad, out_img

    def render(self, img):
        img_undistort = undistort(img, objpoints, imgpoints)
        result = np.copy(img_undistort)

        height, weight = img.shape[0], img.shape[1]
        src = [(570, 470), (722, 470), (1110, 720), (220, 720)]
        dst = [(320, 0), (920, 0), (920, 720), (320, 720)]

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
        left_fit, right_fit, left_curverad, right_curverad, img_poly = self.fit_polynomial(result)

        # step 6: draw lane
        result = draw_lane(img_undistort, result, left_fit, right_fit, Minv)

        # step 7: draw pipeline visualizations
        img_overlay1 = cv2.resize(img_poly, (256, 144))
        img_overlay2 = cv2.resize(cv2.cvtColor(255 * img_sobel, cv2.COLOR_GRAY2RGB), (256, 144))
        result = overlay(result, img_overlay1, x_offset=50, y_offset=50)
        result = overlay(result, img_overlay2, x_offset=316, y_offset=50)
        text = 'Left Curvature: %.3f Right Curvature: %.3f' % (left_curverad, right_curverad)
        result = put_text(result, text, x_offset=50, y_offset=266)

        return result


if __name__ == '__main__':

    line = Lane()

    def pipeline(img):
        try:
            result = line.render(img)
        except Exception as e:
            logger.exception(e)
            result = img
        return result

    #clip1 = VideoFileClip("project_video.mp4")
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(pipeline)
    #white_clip = clip1.fl_image(lambda img: img)
    white_clip.write_videofile("output_videos/project_video.mp4", audio=False)
