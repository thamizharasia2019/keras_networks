

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

import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import ResNet50


from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#ap.add_argument("-a", "--augment", type=int, default=-1,
#	help="whether or not 'on the fly' data augmentation should be used")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
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
	image = cv2.imread(imagePath,1)
	image = cv2.resize(image, (224, 224))
 
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
 
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
base_model=ResNet50(weights='imagenet',include_top=False,input_shape= (img_height,img_width,3)) #imports the mobilenet model and discards the 

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

#for layer in base_model.layers[:-3]:
#    layer.trainable = False

# Check the trainable status of the individual layers

#for layer in base_model.layers:
#    print(layer, layer.trainable)


# ## Added three dense layers and the last layer is having 7 classes



x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation


# ## created new model using base model input and output with 7 classes

model=Model(inputs=base_model.input,outputs=preds)

# ## Displayed model details

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
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',f1])

# hyperparameter tuning

filepath="Augment binary LR 0.0001 data-100 weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10)
reduce1 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

callbacks_list = [checkpoint,early,reduce1]

# data augmentation

train_datagen = ImageDataGenerator(
       
       # preprocessing_function=preprocess_input,
        
        adaptive_equalization=True, 
        histogram_equalization=True,
        rotation_range=90,
        brightness_range=[0.5,2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


val_datagen = ImageDataGenerator(
            
             # preprocessing_function=preprocess_input
               )


validation_generator = val_datagen.flow(testX, testY,batch_size=10)

#INIT_LR = 1e-1
BS = 32
EPOCHS = 20

print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(
	train_datagen.flow(trainX, trainY, batch_size=BS),
	validation_data=validation_generator, 
	steps_per_epoch=len(trainX) // BS,
        callbacks=callbacks_list, 
        verbose=1,
      	epochs=EPOCHS)
# ## Displaying plot of Accuracy Vs epochs


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=1)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])



import matplotlib
from matplotlib import pyplot as plt
results1=H.history
training_accuracy=results1['accuracy']
val_acc=results1['val_accuracy']
epochs1=range(1,len(training_accuracy)+1)
plt.plot(epochs1,training_accuracy,label='Training Accuracy',marker="*",color='r')
plt.plot(epochs1,val_acc,label='Validation Accuracy',marker="+",color='g')
plt.title('Accuracy Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


# ## Displaying plot of Loss Vs epochs

# In[19]:


trainloss=results1['loss']
valloss=results1['val_loss']
epochs1=range(1,len(trainloss)+1)
plt.plot(epochs1,trainloss,label='Training Loss',marker="*",color='r')
plt.plot(epochs1,valloss,label='Validation Loss',marker="+",color='g')
plt.title('Loss Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# ## Displaying plot of F1 score Vs epochs

# In[20]:


trainf1=results1['f1']
valf1=results1['val_f1']
epochs1=range(1,len(trainf1)+1)
plt.plot(epochs1,trainf1,label='Training F1 score',marker="*",color='r')
plt.plot(epochs1,valf1,label='Validation F1 score',marker="+",color='g')
plt.title('F1 score Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 score')


# In[ ]:




