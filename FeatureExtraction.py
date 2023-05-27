from skimage import feature
import numpy as np
import cv2

# ************************* Local binary pattern ************************************

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
    return np.array(hog_windows)

# ************************************ Landmarks ***********************************

def get_landmarks(image, rects, predictor):
    
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.asarray([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


# ************************************ Gabor ***********************************

def build_Gabor_filter(ksize):
        filters = []
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
            return filters

def apply_Gabor_filter(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum