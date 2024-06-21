import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import os
import cv2

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)
    
def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]
    
def create_training_data(categories,datadir,img_size,training_data):
    for category in categories:
        path = os.path.join(datadir, category) # path to categories
        class_num = categories.index(category) 
        for img in os.listdir(path):
            try:
                #img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image and convert to gray scale
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR) #read image as colored
                new_array = cv2.resize(img_array,(img_size,img_size))
                random_rotation(new_array)
                training_data.append([new_array,class_num])
                random_noise(new_array)
                training_data.append([new_array,class_num])
                # horizontal_flip(new_array)
                # training_data.append([new_array,class_num])  
            except Exception as e:
                print("Error creating training data")
                print(e)
                pass
