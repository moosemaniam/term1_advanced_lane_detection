from ops import dir_threshold,mag_thresh,abs_sobel_thresh,apply_thresholding
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


image = cv2.imread("signs_vehicles_xygrad.png")
dir_binary = apply_thresholding(image,ksize=5 )
# Plot the result
cv2.imwrite("output.png",dir_binary)

