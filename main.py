from Preprocessing import Histogram_equalization, Normalization, return_original_image, draw_facebox_crop_face
import matplotlib.pyplot as plt
import cv2
import mtcnn
from FeatureExtraction import LBP, sliding_hog_windows, get_landmarks
import dlib
FILENAME = r"Dataset\train\disgust\Training_96734926.jpg"
# FILENAME = r"C:\Users\sbesrour\Pictures\Camera Roll\WIN_20220715_10_57_11_Pro.jpg"
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
WINDOW_SIZE = 24
WINDOW_STEP = 6

if __name__ == "__main__":

    # read image
    pixels = plt.imread(FILENAME)
    if(len(pixels.shape) < 3):
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

    # Instantiate mtcnn detector and detect faces in image
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(pixels)
    print(faces)

    # Return original image if there is no face detected
    if(len(faces) == 0):
        img = return_original_image(FILENAME)
        
    # Return image with detected face with highest confidence
    else:
        img = draw_facebox_crop_face(FILENAME, faces)

    img = Histogram_equalization(img)
    img = Normalization(img)

    img = cv2.resize(img, (48, 48))
    
    # ******* Feature Extraction *************
    # LBP
    img_lbp = LBP(img, 24, 3)
    # HOG
    
    img_hog = sliding_hog_windows(img, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_STEP, WINDOW_SIZE)
    # Landmarks
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    face_landmarks = get_landmarks(img, face_rects, predictor)

    cv2.imwrite("img_lbp.png", img_lbp)
    