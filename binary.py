

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
import math
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import ResNet50


from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import load_img

from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input


from keras.callbacks import LearningRateScheduler

from keras_radam import RAdam

from pyimagesearch.learningratefinder import LearningRateFinder
#from pyimagesearch import config
from pyimagesearch.clr_callback import CyclicLR


import sys
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#ap.add_argument("-a", "--augment", type=int, default=-1,
#	help="whether or not 'on the fly' data augmentation should be used")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",
#	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# # Using Resnet50 and initialised with weights of imagenet
# ## images in smear 2005 are resized to 224 x224 





# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
 
# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename, load the image, and
	# resize it to be a fixed 64x64 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	#image = load_img(imagePath, target_size=(224, 224))
        # convert the image pixels to a numpy array
	#image = img_to_array(image)
        # reshape data for the model
	#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = cv2.imread(imagePath,1)
	image = cv2.resize(image, (224, 224))
	image= preprocess_input(image)
	# update the data and labels lists, respectively
	data.append(image)
	
	labels.append(label)

data = np.array(data, dtype="float") 
print('loaded data')
#print(len(data))
#print(len(labels))


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
#data = np.array(data, dtype="float") / 255.0
 
# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, random_state=10, shuffle=True)



# Resnet initialisation with imagenet

img_height,img_width = 224,224
num_classes = 2
input_shape= (img_height,img_width,3)
#base_model=ResNet50(weights='imagenet',include_top=False,input_shape= (img_height,img_width,3)) #imports the mobilenet model and discards the 

restnet = ResNet50(include_top=False, weights='imagenet', input_shape= (img_height,img_width,3))


output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)

preds=Dense(num_classes,activation='softmax')(output ) #final layer with softmax activatio

model = Model(inputs=restnet.input, outputs=preds)


# Freeze the layers except the last 4 layers
for layer in restnet.layers[:-3]:
    layer.trainable = False



#model.summary()


# ## Created function to computer F1 SCORE

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ## Compiled model using Adam optimizer and computed accuracy and f1 score

print("[INFO] compiling model...")
#opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)

# optimiser intitialisation



#decay_rate = INIT_LR 
#decay_rate = INIT_LR / EPOCHS 

# learning rate schedule

initial_lrate = 0.1
drop = 0.5
epochs_drop = 10.0

def step_decay(epoch):
	#lrrate = math.floor(initial_lrate/3.0)    
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	#decay_rate.append(lrrate)
	return lrate
 

#learning_rate=0.1
#decay_rate=learning_rate/ 3

#opt = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
# data augmentation

train_datagen = ImageDataGenerator(
       
       # preprocessing_function=preprocess_input,
        
        adaptive_equalization=True, 
        histogram_equalization=True,
        rotation_range=90,
        #brightness_range=[0.5,2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


MIN_LR = 1e-4
MAX_LR = 1e-2
BATCH_SIZE = 8
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 5
CLASSES=['abnormal','normal']


# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_loss.png"])
TRAINING_PLOT_ACC_PATH = os.path.sep.join(["output", "training_acc.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])

opt = keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy',f1])


rate=1
if rate!=1:
	print("[INFO] finding learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(
		train_datagen.flow(trainX, trainY, batch_size=BATCH_SIZE),
		1e-12, 1e+1,epochs=8,
		stepsPerEpoch=np.ceil((len(trainX) / float(BATCH_SIZE))),
		batchSize=BATCH_SIZE)
	 
		# plot the loss for the various learning rates and save the
		# resulting plot to disk
	lrf.plot_loss()
	plt.savefig(LRFIND_PLOT_PATH)
	print("[INFO] learning rate finder complete")
	print("[INFO] examine plot and adjust learning rates before training")
	sys.exit(0)
# hyperparameter tuning


stepSize = STEP_SIZE * (trainX.shape[0] // BATCH_SIZE)
clr = CyclicLR(
	mode=CLR_METHOD,
	base_lr=MIN_LR,
	max_lr=MAX_LR,
	step_size=stepSize)


filepath=" Result binary weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

#early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
reduce1 = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', verbose=2,factor= 0.33, patience=1,min_lr=1e-4)


lrate = LearningRateScheduler(step_decay, verbose=1)


#callbacks_list = [checkpoint,early,reduce1]

#callbacks_list = [checkpoint,reduce1,lrate]

callbacks_list = [checkpoint,reduce1]


#callbacks_list = [checkpoint,lrate]





val_datagen = ImageDataGenerator(
            
             # preprocessing_function=preprocess_input
               )


#validation_generator = val_datagen.flow(testX, testY)



EPOCHS = 30
print("[INFO] training network...")
H = model.fit_generator(
	train_datagen.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BATCH_SIZE,
	epochs=EPOCHS,
	callbacks=callbacks_list,
	verbose=2)



#INIT_LR = 1e-1
#BS = 64


# evaluate the network and show a classification report
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=CLASSES))

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
#plt.plot(N, H.history["accuracy"], label="train_acc")
#plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss ")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(TRAINING_PLOT_PATH)




# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
#plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(TRAINING_PLOT_ACC_PATH)


# plot the learning rate history
#N = np.arange(0, len(clr.history["lr"]))
#plt.figure()
#plt.plot(N, clr.history["lr"])
#plt.title("Cyclical Learning Rate (CLR)")
#plt.xlabel("Training Iterations")
#plt.ylabel("Learning Rate")
#plt.savefig(CLR_PLOT_PATH)

