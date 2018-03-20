import os
import re
import cv2
import importlib.machinery
import numpy as np
import utils
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


def land_detection(image):
    # Load configuration
    loader = importlib.machinery.SourceFileLoader('cf', './config.py')
    cf = loader.load_module()

    # color selection
    color_mask = cv2.inRange(image, np.array(cf.lower_yellow_white), np.array([255, 255, 255]))
    color_select = cv2.bitwise_and(image, image, mask=color_mask)
    grey_image = cv2.cvtColor(color_select, cv2.COLOR_BGR2GRAY)

    # Gaussian smoothing
    image_blurred = cv2.GaussianBlur(grey_image, (cf.guassin_blur_kernel_size, cf.guassin_blur_kernel_size), 0)

    # Define our parameters for Canny and apply
    edges = cv2.Canny(image_blurred, cf.canny_low_threshold, cf.canny_high_threshold)

    # region selection
    vertices = np.array([[x*image.shape[1], y*image.shape[0]] for [x, y] in cf.vertices_ratio], dtype=np.int32)
    masked_image = utils.region_of_interest(edges, np.expand_dims(vertices, axis=0))

    # draw lines on an image given endpoints
    # Hough transform
    lines = cv2.HoughLinesP(masked_image, cf.hough_rho, np.pi/cf.hough_theta_scale, cf.hough_threshold, np.array([]),
                            minLineLength=cf.hough_min_line_length, maxLineGap=cf.hough_max_line_gap)

    line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    utils.draw_lines(cf, line_img, lines)
    # Draw the lines on the edge image
    # initial_img * alpha + img * beta + γ
    lines_edges = cv2.addWeighted(src1=image, alpha=0.8, src2=line_img, beta=1, gamma=0)

    return lines_edges


def main():
    filepath = "test_videos/"
    files = os.listdir(filepath)

    for video_input in files:
        clip = VideoFileClip(os.path.join(filepath, video_input))
        video_output = os.path.join('test_videos_output', video_input.replace('.mp4', '_processed.mp4'))
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ## clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip_out = clip.fl_image(land_detection)  # NOTE: this function expects color images!!
        clip_out.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    main()