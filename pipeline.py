from moviepy.editor import VideoFileClip
import cv2
from cv import *
import numpy as np

objpoints, imgpoints = load_calibration_parameters(6, 9)


def pipeline(img):
    img_undistort = undistort(img, objpoints, imgpoints)
    result = np.copy(img_undistort)

    height, weight = img.shape[0], img.shape[1]
    src = [(570, 470), (722, 470), (1110, 720), (220, 720)]
    dst = [(320, 0), (920, 0), (920, 720), (320, 720)]

    # step 2: gradient and threshold
    result = compose_threshold(result, s_thresh=(170, 255), thresh=(50, 100))

    # step 3: select region of interest
    vertices = np.int32([[
        (src[0][0] - 60, src[0][1]),
        (src[1][0] + 60, src[1][1]),
        (src[2][0] + 60, src[2][1]),
        (src[3][0] - 60, src[3][1])]]
    )
    result = region_of_interest(result, vertices)

    # step 4: perspective transform
    result, M, Minv = unwarp(result, np.float32(src), np.float32(dst))

    # step 5: fit poly
    left_fit, right_fit, img_poly = fit_polynomial(result)

    # step 8: draw lane
    result = draw_lane(img_undistort, result, left_fit, right_fit, Minv)

    return result


if __name__ == '__main__':
    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile("challenge_video_processed.mp4", audio=False)
