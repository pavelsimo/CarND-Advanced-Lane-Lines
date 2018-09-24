from moviepy.editor import VideoFileClip
from cv import *
import numpy as np
import logging

logger = logging.getLogger(__name__)


objpoints, imgpoints = load_calibration_parameters(6, 9)


class Line(object):
    def __init__(self):
        self.prev_left_curverad = -1
        self.prev_right_curverad = -1

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
        left_fit, right_fit, left_curverad, right_curverad, img_poly = fit_polynomial(result)
        if self.prev_left_curverad > 0 and self.prev_right_curverad > 0:
            diff_left_curverad = left_curverad - self.prev_left_curverad
            diff_right_curverad = right_curverad - self.prev_right_curverad
        else:
            diff_left_curverad = 0
            diff_right_curverad = 0

        self.prev_left_curverad = left_curverad
        self.prev_right_curverad = right_curverad

        # step 6: draw lane
        result = draw_lane(img_undistort, result, left_fit, right_fit, Minv)

        # debug
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(result, 'Left Curvature: %.3f Right Curvature: %.3f' % (left_curverad, right_curverad),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        img_overlay1 = cv2.resize(img_poly, (256, 144))
        img_overlay2 = cv2.resize(cv2.cvtColor(255*img_sobel, cv2.COLOR_GRAY2RGB), (256, 144))
        result = overlay(result, img_overlay1, x_offset=50, y_offset=50)
        result = overlay(result, img_overlay2, x_offset=316, y_offset=50)

        # s_img = cv2.resize(img_poly, (256, 144))
        # x_offset = y_offset = 50
        # result[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

        return result


if __name__ == '__main__':

    line = Line()

    def pipeline(img):
        try:
            result = line.render(img)
        except Exception as e:
            logger.exception(e)
            result = img
        return result

    #clip1 = VideoFileClip("project_video.mp4")
    clip1 = VideoFileClip("project_video.mp4").subclip(11, 20)
    white_clip = clip1.fl_image(pipeline)
    #white_clip = clip1.fl_image(lambda img: img)
    white_clip.write_videofile("output_videos/project_video.mp4", audio=False)
