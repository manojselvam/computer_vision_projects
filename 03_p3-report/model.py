import os
import numpy as np
import keras
from keras.layers import Convolution2D, Flatten, Dense, Input, Dropout, Cropping2D, ELU
from keras.models import Sequential
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2, activity_l2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv
import cv2
import scipy.stats as stats
import scipy
import pickle
import pandas
from pandas import read_csv
import pandas as pd
import glob

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle as shuffle


import tensorflow as tf
import csv
import scipy.misc
import pickle
import glob
from pandas import read_csv
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle
import cv2 as cv2


# Load a pickle file
def load_pickle(file_path):
    with open(file_path, mode='rb') as f:
        file_data = pickle.load(f)
        return file_data;


# Create a pickle
def create_pickle(file_path, data):
    pickle.dump(data, open(file_path, "wb"))
    print("Data saved in", file_path)
    
    
def augment_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rand = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * rand
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2BGR)
    return image1

def flip_image(img):
    return cv2.flip(img, 1)


def crop_image(image):
    # crop & scale
    return cv2.resize(image[50:140, :, :], (200, 66))

# [credit] https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.shn2c7jfj
def trans_image(image,steer,trans_range):
    rows, cols, _ = image.shape
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang,tr_x

def preprocess_image(image,angle,augment=False):
    # blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    if augment and np.random.randint(2) == 0:
        image,angle,_ = trans_image(image,angle,125)
    
    img = crop_image(image)
    if augment:
        img = augment_brightness(img)
    
    # format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img,angle

def load_dataset():
    paths = []
    paths.append('./data/driving_log.csv')
    paths.append('./data/sharp_turn.csv')
    paths.append('./data/curves_driving_log.csv')
    paths.append('./data/long_drive.csv')
    data = None
    for path in paths:
        
        df = read_csv(path)
        print(path, len(df))
        if data is None:
            data = df
        else:
            
            data = pandas.concat([data,df],ignore_index=True)
    
    correction = 0.25
    data['left_steer'] = data.steering + correction
    data['right_steer'] = data.steering - correction
    data.center = "./data/" + data.center.str.strip()
    data.left = "./data/" + data.left.str.strip()
    data.right = "./data/" + data.right.str.strip()
    return data

def make_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation="elu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation="elu"))
    model.add(Dropout(.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation="elu"))
    model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(Dropout(.5))
    # nvidia architecture layer has 1164 params at this level. but how ???
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dropout(.25))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse")
    return model

 
def generate_samples(data, augment=True, batch_size = 128):
    
    cameras      = {0:"center",1:"left",2:"right"}
    steer_angles = {0:"steering",1:"left_steer",2:"right_steer"}
    while True:
        # random permutation of indices
        indices = np.random.permutation(data.count()[0])
        
        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            x = []
            y = []
            for i in batch_indices:
                # choose a camera
                camera_idx = 0 
                if augment:
                    camera_idx = np.random.randint(3)
                    
                angle = data[steer_angles[camera_idx]][i]
                
                if(np.abs(angle) > 1.0):
                    camera_idx = 0
                    angle = data[steer_angles[camera_idx]][i]
                    
                image = cv2.imread(data[cameras[camera_idx]][i])
                
                if augment:
                    # toss a coin for train set
                    image,angle = preprocess_image(image,angle,augment=np.random.randint(2) == 0)
                else:
                    image,angle = preprocess_image(image,angle,augment=False)
                    
                   
                # Append to batch
                x.append(image)
                y.append(angle)
            
            # flip half of images in this batch
            for i in range(len(x)):
                if np.abs(y[i]) > 1:
                    y[i] = float(int(y[i]))
                if np.random.randint(2) == 0 and augment:
                    x[i] = flip_image(x[i])
                    y[i] = y[i] * -1

            yield (np.asarray(x), np.asarray(y))
            

def main():
    
    data = load_dataset()
    data = data[data.throttle >= .25]
    #data = data[data.steering != 0]
    data = data.reset_index()

    X_train, X_valid = data,data
    batch_size = 256
 
   
    
    
    train_generator = generate_samples(data, batch_size=batch_size, augment=True)
    validation_generator = generate_samples(data,batch_size=batch_size,augment=False)

    samples_per_epoch = 20480 * 4
        
    samples_per_epoch = int(samples_per_epoch / batch_size) * batch_size
    nb_val_samples = int(len(X_valid) / batch_size) * batch_size

    print(len(X_train), samples_per_epoch)

    
    no_of_epocs = 10
    model = make_model()
    #model.summary()
    
    base_path = './model/steering7/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    model_no = len(glob.glob(base_path + '*.p')) + 1    
        
    checkpoint = ModelCheckpoint(base_path + 'model_' + str(model_no) +'{epoch:02d}.h5')
    # keras does not write histograms when using generators :(
    tensorboard =  keras.callbacks.TensorBoard(log_dir=base_path, histogram_freq=1, write_graph=True, write_images=False)

    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch = samples_per_epoch,
                                         validation_data=validation_generator,
                                         nb_val_samples=nb_val_samples, 
                                         nb_epoch=no_of_epocs, 
                                         callbacks=[checkpoint],
                                         verbose=1)


    
    model.save(str(base_path + 'model_' + str(model_no) + '.h5'))
    create_pickle(str(base_path + 'model_' + str(model_no) + '_history.p'), history_object.history)
    print("Model Saved ")


if __name__ == '__main__':
    main()