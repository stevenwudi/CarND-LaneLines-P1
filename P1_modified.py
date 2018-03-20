import os
import re
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import importlib.machinery
import numpy as np
import utils


def land_detection(image, cf):

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
    filepath = "test_images/"
    pattern = re.compile("^.+processed.jpg$")
    files = os.listdir(filepath)

    # Load configuration
    loader = importlib.machinery.SourceFileLoader('cf', './config.py')
    cf = loader.load_module()

    for file in files:
        # import the image if it is not a saved output
        if not pattern.match(file):
            image_filepath = filepath + file
            image = mpimg.imread(image_filepath)

            # process image
            image_wlines = land_detection(image, cf)
            print('Image ', image_filepath, ' has dimensions: ', image.shape)

            # next image
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            # plot the image
            ax.imshow(image_wlines)
            # writeout the image with "-processed" in the name so it will not be reprocessed.
            plt.savefig(image_filepath.replace(".jpg", "-processed.jpg"))


if __name__ == "__main__":
    main()