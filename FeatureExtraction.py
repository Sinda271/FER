from skimage import feature
import numpy as np

# ************************* Local bunary pattern ************************************

def LBP(image, npoints, radius):
    lbp = feature.local_binary_pattern(image, npoints, radius, method="uniform")
    return lbp

# ******************************** HOG **********************************************

def sliding_hog_windows(image, image_height, image_width, window_step, window_size):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(feature.hog(window, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1)))
    return hog_windows

# ************************************ Landmarks ***********************************

def get_landmarks(image, rects, predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

