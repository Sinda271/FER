from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Preprocessing import Histogram_equalization, Normalization, return_original_image, draw_facebox_crop_face
import mtcnn
from FeatureExtraction import sliding_hog_windows, get_landmarks, build_Gabor_filter, apply_Gabor_filter
import dlib
from PIL import Image
import os


TEST_IMAGE = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\FER2013\test\neutral\PrivateTest_687498.jpg"
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
WINDOW_SIZE = 24
WINDOW_STEP = 6
GABOR_KERNEL_SIZE = 3


def classify(path, imageHeight, imageWidth, windowSize, windowStep, gaborKerSize):

    # Instantiate mtcnn detector
    detector = mtcnn.MTCNN()


    image = Image.open(path)
    image.save(path[:-3] + "png")

    pixels = plt.imread(path[:-3] + "png")

    

    if(len(pixels.shape) < 3):
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)

    faces = detector.detect_faces(pixels)
    print(faces)

    # Return original image if there is no face detected
    if(len(faces) == 0):
        img = return_original_image(path)
        
    # Return image with detected face with highest confidence
    else:
        img = draw_facebox_crop_face(path, faces)

    # Covert images to gray for histogram equalization 
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Histogram_equalization(img)

    img = Normalization(img)

    img = cv2.resize(img, (imageWidth, imageHeight))
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
    model_input = model_input[:-24].reshape(1, 160, 160, 1) # 3203, 4, 2, 1
    print(f"this is the model input shape: {model_input.shape}")

    if os.path.isfile(path[:-3] + "png"):
        os.remove(path[:-3] + "png")

    return model_input
        

if __name__ == "__main__":
    model = keras.models.load_model('model')
    model.summary()

    model_inp = classify(TEST_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_SIZE, WINDOW_STEP, GABOR_KERNEL_SIZE)
    prediction = model.predict(model_inp)
    print(prediction)