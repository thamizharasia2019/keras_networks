#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from keras.applications import ResNet50
# from keras.applications import Xception # TensorFlow ONLY

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--train", required=True,
	help="path to augmented train dataset")
ap.add_argument("-t", "--test", required=True,
	help="path to test dataset")

args = vars(ap.parse_args())
# # Using Resnet50 and initialised with weights of imagenet
# ## images in smear 2005 are resized to 224 x224 



# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading training images...")
imagePaths = list(paths.list_images('./dataset_291119FN/'+args["train"]))
data = []
labels = []
 
i=0
# loop over the image paths

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))


	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (128, 128))
 
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
trainX = np.array(data, dtype="float") / 255.0
 
# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
trainY = np_utils.to_categorical(labels, 2)

#trainX=data
#trainY=labels


print('train data completed')
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading testing images...")
imagePaths1 = list(paths.list_images('./dataset_291119FN/'+args["test"]))
print(len(imagePaths1))
data1 = []
labels1 = []

ii=0
# loop over the image paths

for (ii, imagePath) in enumerate(imagePaths1):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(ii + 1,
		len(imagePaths1)))

	# extract the class label from the filename, load the image, and
	# resize it to be a fixed 64x64 pixels, ignoring aspect ratio
	
	
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (128, 128))
 
	# update the data and labels lists, respectively
	data1.append(image)
	labels1.append(label)


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
testX = np.array(data1, dtype="float") / 255.0
 
# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le1 = LabelEncoder()
labels1 = le1.fit_transform(labels1)
testY = np_utils.to_categorical(labels1, 2)
print('testing completed')


#testX=data
#testY=labels

outfile='./dataset_291119FN/tmp/traintestbinary.npz'
np.savez(outfile, trainX,trainY, testX,testY)

print('file written')



