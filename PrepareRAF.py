
from cProfile import label
from glob import glob
import os
PATH_TO_IMAGES = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\basic\Image\aligned\aligned"
IMAGE_LABELS = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\basic\EmoLabel\list_patition_label.txt"
OUTPUT_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"
EMOTIONS = {
    "surprise" : 1,
    "fear" : 2,
    "disgust" : 3,
    "happy" : 4,
    "sad" : 5,
    "angry" : 6,
    "neutral" : 7
}

# Create emotion folders
for emotion in EMOTIONS.keys():
    path = os.path.join(OUTPUT_FOLDER, emotion)
    if (os.path.exists(path) == False):
        os.mkdir(path)

# Read image labels from txt file
labels = []
f = open(IMAGE_LABELS, "r")
for line in f.readlines():
    t = (line.split(" ")[0], line.split(" ")[1].replace("\n", ""))
    labels.append(t)

# Rename images to the same name in the label file
images = glob(PATH_TO_IMAGES + "\*")
for pth in images:
    new_name = PATH_TO_IMAGES + "\\"+ pth.split("\\")[-1].replace("_aligned", "")
    os.rename(pth, new_name)

# Put images in the right folder
for tuple in labels:
    for path in images:
        if(tuple[0] in path):
           
            if(tuple[1] == "1"):
                os.rename(path, OUTPUT_FOLDER + r"\surprise\\" + path.split("\\")[-1])
            elif(tuple[1] == "2"):
                os.rename(path, OUTPUT_FOLDER + r"\fear\\" + path.split("\\")[-1])
            elif(tuple[1] == "3"):
                os.rename(path, OUTPUT_FOLDER + r"\disgust\\" + path.split("\\")[-1])
            elif(tuple[1] == "4"):
                os.rename(path, OUTPUT_FOLDER + r"\happy\\" + path.split("\\")[-1])
            elif(tuple[1] == "5"):
                os.rename(path, OUTPUT_FOLDER + r"\sad\\" + path.split("\\")[-1])
            elif(tuple[1] == "6"):
                os.rename(path, OUTPUT_FOLDER + r"\angry\\" + path.split("\\")[-1])
            elif(tuple[1] == "7"):
                os.rename(path, OUTPUT_FOLDER + r"\neutral\\" + path.split("\\")[-1])
            