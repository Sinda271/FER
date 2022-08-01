from PrepareModelInput import build_model_input

DATASET_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
WINDOW_SIZE = 24
WINDOW_STEP = 6
GABOR_KERNEL_SIZE = 3

if __name__ == "__main__":

    # Features
    angry, surprise, fear, disgust, happy, sad, neutral = build_model_input(DATASET_FOLDER, IMAGE_HEIGHT, IMAGE_WIDTH, WINDOW_SIZE, WINDOW_STEP, GABOR_KERNEL_SIZE)
    # Labels
    angry_label = [1 for i in range(len(angry))]
    surprise_label = [2 for i in range(len(surprise))]
    fear_label = [3 for i in range(len(fear))]
    disgust_label = [4 for i in range(len(disgust))]
    happy_label = [5 for i in range(len(happy))]
    sad_label = [6 for i in range(len(sad))]
    neutral_label = [7 for i in range(len(neutral))]
