from glob import glob
import numpy as np
from Preprocessing import Histogram_equalization, Normalization, return_original_image, draw_facebox_crop_face
import matplotlib.pyplot as plt
import cv2
import mtcnn
from FeatureExtraction import LBP, sliding_hog_windows, get_landmarks, build_Gabor_filter, apply_Gabor_filter
import dlib
from pathlib import Path


DATASET_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
WINDOW_SIZE = 24
WINDOW_STEP = 6
GABOR_KERNEL_SIZE = 3

def build_model_input(datasetPath, imageHeight, imageWidth, windowSize, windowStep, gaborKerSize):

    angry_inputs = []
    surprise_inputs = []
    fear_inputs = []
    disgust_inputs = []
    happy_inputs = []
    sad_inputs = []
    neutral_inputs = []
    images = glob(datasetPath + "\*\*")
    # Instantiate mtcnn detector
    detector = mtcnn.MTCNN()

    for path in images:
        # read image
        print(path)
        pth = Path(path)
        print("***************************** ", pth.parent.name)
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = Histogram_equalization(img)

        img = Normalization(img)

        img = cv2.resize(img, (imageWidth, imageHeight))
        
        # ******* Feature Extraction *************

        # LBP
        # img_lbp = LBP(img, 24, 3)

        # HOG
        img_hog = sliding_hog_windows(img, imageHeight, imageWidth, windowStep, windowSize)
        print(f"this is the HOG output shape: {img_hog.shape}")

        # Landmarks
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
        face_landmarks = get_landmarks(img, face_rects, predictor)
        print(f"this is the landmarks output shape: {face_landmarks.shape}")

        # Gabor
        gabor_filter = build_Gabor_filter(gaborKerSize)
        img_gabor = apply_Gabor_filter(img, gabor_filter)
        print(f"this is the GABOR output shape: {img_gabor.shape}")

        # combine hog, landmarks and gabor to represent one image
        model_input = np.concatenate((img_hog.flatten(), face_landmarks.flatten(), img_gabor.flatten()))
        model_input = model_input[:-24].reshape(160, 160, 1) # 3203, 4, 2, 1
        print(f"this is the model input shape: {model_input.shape}")

        # put each model input in the appropriate array
        if(pth.parent.name == "angry"):
            angry_inputs.append(model_input)
        elif(pth.parent.name == "surprise"):
            surprise_inputs.append(model_input)
        elif(pth.parent.name == "fear"):
            fear_inputs.append(model_input)
        elif(pth.parent.name == "disgust"):
            disgust_inputs.append(model_input)
        elif(pth.parent.name == "happy"):
            happy_inputs.append(model_input)
        elif(pth.parent.name == "sad"):
            sad_inputs.append(model_input)
        elif(pth.parent.name == "neutral"):
            neutral_inputs.append(model_input)
       
    return np.array(angry_inputs), np.array(surprise_inputs), np.array(fear_inputs), np.array(disgust_inputs), np.array(happy_inputs), np.array(sad_inputs), np.array(neutral_inputs) 
    # cv2.imwrite("img_gabor.png", img_gabor)

if __name__ == "__main__":
    angry, surprise, fear, disgust, happy, sad, neutral = build_model_input(DATASET_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_SIZE, WINDOW_STEP, GABOR_KERNEL_SIZE)