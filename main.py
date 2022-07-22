from glob import glob
from Preprocessing import Histogram_equalization, Normalization, return_original_image, draw_facebox_crop_face
import matplotlib.pyplot as plt
import cv2
import mtcnn
from FeatureExtraction import LBP, sliding_hog_windows, get_landmarks, build_Gabor_filter, apply_Gabor_filter
import dlib

# FILENAME = r"Dataset\FER2013\train\disgust\Training_96734926.jpg"
# FILENAME = r"C:\Users\sbesrour\Pictures\Camera Roll\WIN_20220715_10_57_11_Pro.jpg"
DATASET_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"

IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
WINDOW_SIZE = 24
WINDOW_STEP = 6
GABOR_KERNEL_SIZE = 3

if __name__ == "__main__":

    images = glob(DATASET_FOLDER + "\*\*")
    # Instantiate mtcnn detector
    detector = mtcnn.MTCNN()

    for path in images:
        # read image
        print(path)
        pixels = plt.imread(path)
        if(len(pixels.shape) < 3):
            pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

        # Detect faces in image
        faces = detector.detect_faces(pixels)
        print(faces)

        # Return original image if there is no face detected
        if(len(faces) == 0):
            img = return_original_image(path)
            
        # Return image with detected face with highest confidence
        else:
            img = draw_facebox_crop_face(path, faces)

        # Covert images to gray for histogram equalization  
        img = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        img = Histogram_equalization(img)
        img = Normalization(img)

        img = cv2.resize(img, (48, 48))
        
        # ******* Feature Extraction *************

        # LBP
        # img_lbp = LBP(img, 24, 3)

        # HOG
        img_hog = sliding_hog_windows(img, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_STEP, WINDOW_SIZE)

        # Landmarks
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(img, face_rects, predictor)

        # Gabor
        gabor_filter = build_Gabor_filter(GABOR_KERNEL_SIZE)
        img_gabor = apply_Gabor_filter(img, gabor_filter)


    # cv2.imwrite("img_gabor.png", img_gabor)
    