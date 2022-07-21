import cv2
import mtcnn
import matplotlib.pyplot as plt
import numpy as np

def get_max_confidence(mtcnn_results):
    confidence = []
    for res in mtcnn_results:
        confidence.append(res['confidence'])
    return max(confidence)


def draw_facebox_crop_face(filename, result_list):

    max_confidence = get_max_confidence(result_list)
    # load the image
    data = plt.imread(filename)
    # get the context for drawing boxes
    # ax = plt.gca()
    # plot each box
    for result in result_list:
        # get result with highest confidence
        if(result['confidence'] == max_confidence):
            # get coordinates
            x, y, width, height = result['box']
            # create the shape
            rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
            # draw the box
            # ax.add_patch(rect)
            cropped_img = data[y:y+height, x:x+width]
            
    return cropped_img


def return_original_image(filename):
    # load the image
    data = plt.imread(filename)
    return data

def Histogram_equalization(image):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    return cl1

def Normalization(image):
    norm_img = np.zeros((24,24))
    norm_img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)
    return norm_img


if __name__ == "__main__":

    # read image
    FILENAME = r"Dataset\train\disgust\Training_96734926.jpg"
    # FILENAME = r"C:\Users\sbesrour\Pictures\Camera Roll\WIN_20220715_10_57_11_Pro.jpg"
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
    # cv2.imshow('image',img)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite("img.png", img)
    
