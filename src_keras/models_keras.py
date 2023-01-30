import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D,UpSampling2D, UpSampling3D, Convolution3D, MaxPooling3D



"""The Laplacian of Gaussian"""
def LoG_Blob_Detector():
    model = keras.Sequential()
    model.add(Convolution2D(1, (3, 3), activation='relu', padding='same', input_shape=(height, width, channels)))
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(1, (3, 3), activation='sigmoid', padding='same'))

    return model


"""Difference of Gaussians"""
def DoG_Blob_Detector():
    model = keras.Sequential()
    model.add(Convolution2D(filters=1, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(height, width, channels)))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(Convolution2D(filters=1, kernel_size=(3, 3), strides=1, activation='sigmoid'))

    return model


"""The determinant of the Hessian"""
def DoH_blob_detector():
    model = keras.Sequential()
    model.add(Convolution2D(filters=8, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(height, width, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    return model


"""The Laplacian of Gaussian for 3D"""
def LoG_Blob_Detector_3D():
    model = keras.Sequential()
    model.add(Convolution3D(1, (3, 3, 3), activation='relu', padding='same', input_shape=(32, 32, 32, channels)))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(1, (3, 3, 3), activation='sigmoid', padding='same'))

    return model


"""Difference of Gaussians for 3D"""
def DoG_Blob_Detector_3D():
    model = keras.Sequential()
    model.add(Convolution3D(filters=1, kernel_size=(3, 3, 3), strides=1, activation='relu', input_shape=(depth, height, width, channels)))
    model.add(Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
    model.add(Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Convolution3D(filters=64, kernel_size=(3, 3, 3), strides=1, activation='relu'))
    model.add(UpSampling3D((2, 2, 2)))
    model.add(Convolution3D(filters=32, kernel_size=(3, 3, 3), strides=1, activation='relu'))
    model.add(Convolution3D(filters=1, kernel_size=(3, 3, 3), strides=1, activation='sigmoid'))

    return model


"""The determinant of the Hessian"""
def DoH_blob_detector_3D():
    model = keras.Sequential()
    model.add(Convolution3D(filters=8, kernel_size=(5, 5, 5), strides=1, activation='relu', input_shape=(depth, height, width, channels)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
    model.add(Convolution3D(filters=16, kernel_size=(5, 5, 5),strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    return model
