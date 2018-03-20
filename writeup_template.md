# **Finding Lane Lines on the Road** 



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[image_1]: ./examples/1.png
[image_2]: ./examples/2.png 
[image_3]: ./examples/3.png 
[image_4]: ./examples/4.png 
[image_5]: ./examples/5.png 

---

### Reflection

### 1. Describe the pipeline:

(1) Color selection with rgb_threshold set as: [192, 192, 32]

![alt_text][image_1]

(2) Gaussian blur an image with kernel size of 7

![alt_text][image_2]


(3) Canny Edgy detection with threshold set within the range [10, 30]

![alt_text][image_3]

(4) Roi selection with a trapezoid:

![alt_text][image_4]

(5) Hough transform and plot the detected lanes overlayed with the original image:

![alt text][image_5]


As part of the description, explain the modification in the utils.draw_lines() function include:

(1) decide the slope for the detected lines by Canny edgy detector, specifically:
the left lane slope should be within the range [-1.0, -0.5] and right lane slope [0.5, 1.0]

(2) Then we average the lanes


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lanes are curved because currently the Hough 
transform can only detect straight lines

Another shortcoming could be the the lane colors are not within the color range defined. 
(This can be caused due to low light settings, weather conditions, etc.)


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to using advanced lane curve detection technique introduced in the later module.

Another potential improvement could be to use Fully Convolutional Network as a semantic segmentation task.
