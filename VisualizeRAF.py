import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
DATASET_FOLDER = r"C:\Users\sbesrour\Desktop\personal\fer\Dataset\RAF\output"

def data_distribution_per_class(dataset_path):

    count = []
    for root_dir, cur_dir, files in os.walk(dataset_path):
        count.append(len(files))
    
    return os.listdir(dataset_path), count[1:]
        

if __name__ == "__main__":
    x, y = data_distribution_per_class(DATASET_FOLDER)
    print(y)
    print(x)
    plt.bar(x, y)
    plt.show()