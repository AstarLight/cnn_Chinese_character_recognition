#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定使用的GPU的ID

import numpy as np
import tensorflow as tf
import cv2
import argparse
from keras.backend.tensorflow_backend import set_session
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model  
#from keras_train import load_dataset
K.set_image_dim_ordering('th')


def load_image(image_path, norm_shape=(28, 28)):
    image = cv2.imread(image_path, 0)
    resized_img = cv2.resize(image, norm_shape,
                             interpolation=cv2.INTER_CUBIC)
    res,img= cv2.threshold(resized_img, 120, 255, cv2.THRESH_BINARY);
    cv2.imwrite('binary.jpg',img)
    return img
 
def load_dataset(image_label_path,prefix,norm_shape=(28, 28)):
    images = []
    labels = []
    for line in open(prefix+image_label_path, "r"):
        words = line.split()
        if len(words) < 2:
            continue
        image_path = words[0]
        label = words[1]
        images.append(load_image(prefix+image_path))
        labels.append(int(label))
    images = np.asarray(images, dtype=np.float32)
    images = images / 255.0
    max_n_label = max(labels) + 1
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels, max_n_label

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Keras CNN OCR predict demo')

    parser.add_argument('--json_path',dest='json_path',help='json path',
                        default=None, required=True)

    parser.add_argument('--model_path',dest='model_path',help='model path',
                        default=None, required=True)
                      
    parser.add_argument('--test_path',dest='test_path',help='test images path',
                        default=None, required=True)                        
    args = parser.parse_args()

    return args

    
args = parse_args()
NET_JSON_PATH = args.json_path
MODEL_PATH = args.model_path
TEST_PATH = args.test_path + '/'

max_n_label = 16

def read_model(cross=''):
    json_name = NET_JSON_PATH
    weight_name = MODEL_PATH
    model = model_from_json(open(json_name).read())
    model.load_weights(weight_name)
    return model

model = load_model(MODEL_PATH)

X_test, y_test, _ = load_dataset("predict.txt",TEST_PATH)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
Y_test = np_utils.to_categorical(y_test, max_n_label)

score = model.evaluate(X_test, Y_test, verbose=1)
print '\n',model.metrics_names,' :',score
