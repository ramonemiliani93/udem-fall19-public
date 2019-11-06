import cv2
import numpy as np
import rospy
from scipy.signal import convolve2d
## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
def inRange(hsv_image, low_range, high_range):
	lower = (hsv_image >= low_range.reshape(1, 1, 3)).all(axis=-1)
	upper = (hsv_image <= high_range.reshape(1, 1, 3)).all(axis=-1)
	return np.uint8(255 * lower * upper)


def bitwise_or(bitwise1, bitwise2):
	return np.uint8((bitwise1 + bitwise2) >= 0.5)


def bitwise_and(bitwise1, bitwise2):
	return bitwise1 * bitwise2


def getStructuringElement(shape, size):
	if shape == 0:
		structuring_element = np.ones(size, dtype='uint8')
	elif shape == 1:
		rows, columns = size
		structuring_element = np.zeros(size, dtype='uint8')
		structuring_element[rows // 2, :] = 1
		structuring_element[:, columns // 2] = 1
	else:
		rows, columns = size
		structuring_element = np.ones((rows - 2, columns - 2), dtype='uint8')
		structuring_element = np.pad(structuring_element, 1, 'constant', constant_values=0)
		structuring_element[0, columns // 2] = 1
		structuring_element[-1, columns // 2] = 1
		structuring_element[rows // 3:-(rows // 3), 0] = 1
		structuring_element[rows // 3:-(rows // 3), -1] = 1

	return structuring_element


def dilate(bitwise, kernel):
	kernel = np.flip(np.flip(kernel, axis=0), axis=1)
	return np.uint8(convolve2d(bitwise, kernel, 'same') > 0)


## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)