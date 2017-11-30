#coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #指定使用的GPU的ID

import numpy as np
import tensorflow as tf
import cv2
import argparse
from keras.backend.tensorflow_backend import set_session
 
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


def load_image(image_path, norm_shape=(28, 28)):
    image = cv2.imread(image_path, 0)
    resized_img = cv2.resize(image, norm_shape,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img
 
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
    parser = argparse.ArgumentParser(description='Keras CNN OCR demo')

    parser.add_argument('--train_path',dest='train_path',help='training data path',
                        default=None, required=True)

    parser.add_argument('--model_path',dest='model_path',help='model output path',
                        default=None, required=True)
    args = parser.parse_args()

    return args

    
args = parse_args()
#print type(args.train_path)
TRAINING_PATH = args.train_path + '/'
MODEL_PATH = args.model_path + '/' 

 
X_train, y_train, max_n_label = load_dataset("train.txt",TRAINING_PATH)
X_test, y_test, _ = load_dataset("test.txt",TRAINING_PATH)
 
# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
 
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, max_n_label)
Y_test = np_utils.to_categorical(y_test, max_n_label)


# 7. Define model architecture
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu',
                        input_shape=(1,28,28), dim_ordering="th"))
model.add(Convolution2D(32, (3, 3), activation='relu', dim_ordering="th"))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(max_n_label, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              

# 9. Fit model on training data
model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

#10. save model
model.save(MODEL_PATH + 'model_ocr.h5')
#json_string = model.to_json()  #等价于 json_string = model.get_config()  
#open(MODEL_PATH + 'model_architecture.json','w+').write(json_string)   
#model.save_weights(MODEL_PATH + 'model_weights.h5')

# 11. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)
print '\n',model.metrics_names,' :',score

