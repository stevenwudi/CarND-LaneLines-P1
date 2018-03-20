import importlib.machinery
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import cv2
import utils


def main():

    # Load configuration
    loader = importlib.machinery.SourceFileLoader('cf', './config.py')
    cf = loader.load_module()

    # reading in an image
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    print('This image is:', type(image), 'with dimensions:', image.shape)

    # region selection
    vertices = np.array([[x*image.shape[1], y*image.shape[0]] for [x, y] in cf.vertices_ratio], dtype=np.int32)
    masked_image = utils.region_of_interest(image, np.expand_dims(vertices, axis=0))
    polygon = patches.Polygon(vertices, linewidth=2, edgecolor='r', facecolor='none')
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Add the patch to the Axes
    # Display the image
    ax.imshow(image)
    ax.add_patch(polygon)

    # Gaussian smoothing
    image_blurred = cv2.GaussianBlur(masked_image, (cf.guassin_blur_kernel_size, cf.guassin_blur_kernel_size), 0)
    # color selection
    color_mask = cv2.inRange(image_blurred, np.array(cf.rgb_threshold), np.array([255, 255, 255]))
    color_select = cv2.bitwise_and(image, image, mask=color_mask)

    plt.figure()
    plt.imshow(color_select, cmap='gray')

    # Define our parameters for Canny and apply
    edges = cv2.Canny(color_select, cf.canny_low_threshold, cf.canny_high_threshold)
    plt.figure()
    plt.imshow(edges, cmap='gray')

    # draw lines on an image given endpoints
    # Hough transform
    lines = cv2.HoughLinesP(edges, cf.hough_rho, np.pi/cf.hough_theta_scale, cf.hough_threshold, np.array([]),
                            minLineLength=cf.hough_min_line_length, maxLineGap=cf.hough_max_line_gap)

    line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    utils.draw_lines(cf, line_img, lines)
    # Draw the lines on the edge image
    # initial_img * alpha + img * beta + γ
    lines_edges = cv2.addWeighted(src1=image, alpha=0.8, src2=line_img, beta=1, gamma=0)
    plt.figure()
    plt.imshow(lines_edges)
    plt.waitforbuttonpress()

    print('Done')


if __name__ == "__main__":
    main()