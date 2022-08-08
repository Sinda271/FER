from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

import tensorflow as tf

NUM_CLASSES = 7

def model(input_shape, num_classes):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])

    return model

def fix_to_categorical(y_train):
    liste=[]
    for item in tf.keras.utils.to_categorical(y_train):
        item=np.delete(item,0)
        liste.append(item)
    y_train=np.array(liste)
    return y_train


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
BATCH_SIZE = 2
def build_model_input(datasetPath, imageHeight, imageWidth, windowSize, windowStep, gaborKerSize):

    angry_inputs = []

    images = glob(datasetPath + "\*\*")
    # Instantiate mtcnn detector
    detector = mtcnn.MTCNN()

    for path in images[:10]:
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
        # model_input = torch.cat((torch.flatten(img_hog), torch.flatten(face_landmarks), torch.flatten(img_gabor)))
        # model_input = np.array([img_hog, face_landmarks, img_gabor])
        model_input = np.concatenate((img_hog.flatten(), face_landmarks.flatten(), img_gabor.flatten()))
        model_input = model_input[:-24].reshape(160, 160, 1) # 3203, 4, 2, 1
        # print(model_input)
        # model_input = np.reshape(model_input, (3, 1))
        # model_input = np.array(model_input, dtype=np.float32)
        print(f"this is the model input shape: {model_input.shape}")

        # put each model input in the appropriate array
        
        angry_inputs.append(model_input)


       
    return np.asarray(angry_inputs)




if __name__ == "__main__":

    neutral = build_model_input(DATASET_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_SIZE, WINDOW_STEP, GABOR_KERNEL_SIZE)
    neutral_label = np.asarray([7 for i in range(len(neutral))], dtype=np.int32)
    # neutral_label = tf.keras.utils.to_categorical(neutral_label)
    neutral_label = fix_to_categorical(neutral_label)
    print(neutral_label)
   
    model = model(input_shape=(160, 160, 1), num_classes=NUM_CLASSES)
    model.fit(neutral, neutral_label, batch_size = 1, epochs = 6)
    model.save('model')
        

    
