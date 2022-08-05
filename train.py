from PrepareModelInput import build_model_input
import numpy as np
from model import model
import tensorflow as tf

DATASET_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
WINDOW_SIZE = 24
WINDOW_STEP = 6
GABOR_KERNEL_SIZE = 3
NUM_CLASSES = 7
BATCH_SIZE = 16
NUM_EPOCHS = 10


if __name__ == "__main__":

    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
    with tf.device("/GPU:0"):
        # Features
        angry, surprise, fear, disgust, happy, sad, neutral = build_model_input(DATASET_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_SIZE, WINDOW_STEP, GABOR_KERNEL_SIZE)
        # Labels
        angry_label = np.array([1 for i in range(len(angry))])
        surprise_label = np.array([2 for i in range(len(surprise))])
        fear_label = np.array([3 for i in range(len(fear))])
        disgust_label = np.array([4 for i in range(len(disgust))])
        happy_label = np.array([5 for i in range(len(happy))])
        sad_label = np.array([6 for i in range(len(sad))])
        neutral_label = np.array([7 for i in range(len(neutral))])
        
        X_train = np.concatenate((angry, surprise, fear, disgust, happy, sad, neutral))
        y_train = np.concatenate((angry_label, surprise_label, fear_label, disgust_label, happy_label, sad_label, neutral_label))

        print(f"Features shape: {X_train}   labels shape: {y_train}")

        # Model
        model = model(input_shape=(160, 160, 1), num_classes=NUM_CLASSES)
        model.fit(X_train, y_train, batch_size = 32, epochs = 100)
        model.save('model')
    

