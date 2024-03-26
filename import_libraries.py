import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('dark_background')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
