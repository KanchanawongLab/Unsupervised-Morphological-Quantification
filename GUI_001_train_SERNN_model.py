import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from imgaug import augmenters as iaa

from SERNN_model import seq_model

import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Load images into 4D numpy arrays
def load_image(image_dir):
  image_list = []
  num_img = 0
  for filename in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, filename))
    if img is not None:
      img = np.float16(img)/np.max(img)
      img = np.uint8(img*255)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.float16(img)/np.max(img)
      image_list.append(img)
      num_img += 1
  print("\nImage list shape = ", np.shape(image_list))
  
  image_np = np.empty((num_img, 256, 256, 3))

  for i in range(num_img):
    image_np[i,:,:,:] = image_list[i]

  return image_np

#Load raw datasets
print("\nLoading images...")
cell_1_read = load_image("full data/MDCK_C1/")
cell_2_read = load_image("full data/MDCK_C4/")
cell_3_read = load_image("full data/MDCK_FN/")
cell_4_read = load_image("full data/MDCK_LN/")
cell_5_read = load_image("full data/MDCK_LN10/")
cell_6_read = load_image("full data/MDCK_VN/")
cell_7_read = load_image("full data/MEF_C1/")
cell_8_read = load_image("full data/MEF_C4/")
cell_9_read = load_image("full data/MEF_FN/")
cell_10_read = load_image("full data/MEF_LN/")
cell_11_read = load_image("full data/MEF_LN10/")
cell_12_read = load_image("full data/MEF_VN/")

#Each class has a total of 80 images
#We will use the first 50 images for augmentation and training
cell_1 = cell_1_read[:50]
cell_2 = cell_2_read[:50]
cell_3 = cell_3_read[:50]
cell_4 = cell_4_read[:50]
cell_5 = cell_5_read[:50]
cell_6 = cell_6_read[:50]
cell_7 = cell_7_read[:50]
cell_8 = cell_8_read[:50]
cell_9 = cell_9_read[:50]
cell_10 = cell_10_read[:50]
cell_11 = cell_11_read[:50]
cell_12 = cell_12_read[:50]

#Perform augmentation via iaa
def img_aug(img_stack):
  #Define augmentation functions
  # Geometric augmentations
  flip_lr = iaa.Sequential([iaa.Fliplr(1.0)])
  flip_ud = iaa.Sequential([iaa.Flipud(1.0)])
  scale_small = iaa.Sequential([iaa.Affine(scale = (0.8, 0.95))])
  scale_large = iaa.Sequential([iaa.Affine(scale = (1.05, 1.2))])
  rotate_45 = iaa.Sequential([iaa.Affine(rotate = 45)])
  rotate_90 = iaa.Sequential([iaa.Affine(rotate = 90)])
  rotate_135 = iaa.Sequential([iaa.Affine(rotate = 135)])
  rotate_180 = iaa.Sequential([iaa.Affine(rotate = 180)])
  rotate_225 = iaa.Sequential([iaa.Affine(rotate = 225)])
  rotate_270 = iaa.Sequential([iaa.Affine(rotate = 270)])
  rotate_315 = iaa.Sequential([iaa.Affine(rotate = 315)])
  # Arithemtic augmentation
  gaussian_noise = iaa.Sequential([iaa.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05*255))])
  multiply = iaa.Sequential([iaa.Multiply((0.8, 1.2))])
  dropout = iaa.Sequential([iaa.Dropout(p = (0, 0.3))])
  salt_pepper = iaa.Sequential([iaa.SaltAndPepper(0.1)])

  #Create empty arrays
  cell_lr = np.empty((len(img_stack), 256, 256, 3))
  cell_ud = np.empty((len(img_stack), 256, 256, 3))
  cell_ss = np.empty((len(img_stack), 256, 256, 3))
  cell_sl = np.empty((len(img_stack), 256, 256, 3))
  cell_45 = np.empty((len(img_stack), 256, 256, 3))
  cell_90 = np.empty((len(img_stack), 256, 256, 3))
  cell_135 = np.empty((len(img_stack), 256, 256, 3))
  cell_180 = np.empty((len(img_stack), 256, 256, 3))
  cell_225 = np.empty((len(img_stack), 256, 256, 3))
  cell_270 = np.empty((len(img_stack), 256, 256, 3))
  cell_315 = np.empty((len(img_stack), 256, 256, 3))
  cell_noise = np.empty((len(img_stack), 256, 256, 3))
  cell_multiply = np.empty((len(img_stack), 256, 256, 3))
  cell_dropout = np.empty((len(img_stack), 256, 256, 3))
  cell_saltpepper = np.empty((len(img_stack), 256, 256, 3))

  #Perform augmentation
  for i in range(len(img_stack)):
    cell_lr[i] = flip_lr(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_ud[i] = flip_ud(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_ss[i] = scale_small(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_sl[i] = scale_large(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_45[i] = rotate_45(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_90[i] = rotate_90(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_135[i] = rotate_135(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_180[i] = rotate_180(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_225[i] = rotate_225(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_270[i] = rotate_270(image = img_stack[i])

  for i in range(len(img_stack)):
    cell_315[i] = rotate_315(image = img_stack[i])

  for i in range(len(img_stack)):
    img = img_stack[i]
    img = np.uint8(img*255)
    noise_img = gaussian_noise(image = img)
    noise_img = np.float16(noise_img)/np.max(noise_img)
    cell_noise[i] = noise_img

  for i in range(len(img_stack)):
    img = img_stack[i]
    img = np.uint8(img*255)
    noise_img = multiply(image = img)
    noise_img = np.float16(noise_img)/np.max(noise_img)
    cell_multiply[i] = noise_img

  for i in range(len(img_stack)):
    img = img_stack[i]
    img = np.uint8(img*255)
    noise_img = dropout(image = img)
    noise_img = np.float16(noise_img)/np.max(noise_img)
    cell_dropout[i] = noise_img

  for i in range(len(img_stack)):
    img = img_stack[i]
    img = np.uint8(img*255)
    saltpepper_img = salt_pepper(image = img)
    saltpepper_img = np.float16(saltpepper_img)/np.max(saltpepper_img)
    cell_saltpepper[i] = saltpepper_img

  cell_stack = np.vstack((cell_lr, cell_ud, cell_ss, cell_sl, cell_45, cell_90, cell_135, cell_180, cell_225, cell_270, cell_315, 
    cell_noise, cell_multiply, cell_dropout, cell_saltpepper))

  print("\nAugmented cell stack shape = ", np.shape(cell_stack))
  print("Memory size of array: {} GB".format(float(cell_stack.nbytes/1000000000)))

  return cell_stack

#Create augmented datasets
cell_1_aug = img_aug(cell_1)
cell_2_aug = img_aug(cell_2)
cell_3_aug = img_aug(cell_3)
cell_4_aug = img_aug(cell_4)
cell_5_aug = img_aug(cell_5)
cell_6_aug = img_aug(cell_6)
cell_7_aug = img_aug(cell_7)
cell_8_aug = img_aug(cell_8)
cell_9_aug = img_aug(cell_9)
cell_10_aug = img_aug(cell_10)
cell_11_aug = img_aug(cell_11)
cell_12_aug = img_aug(cell_12)

def img_aug_label(img_stack, num_label):
  cell_label = np.empty((len(img_stack)))
  initial = 0
  for i in range(len(cell_label)):
    cell_label[i] = num_label

  return cell_label

#Create training labels
cell_1_aug_label = img_aug_label(cell_1_aug, 0)
cell_2_aug_label = img_aug_label(cell_2_aug, 1)
cell_3_aug_label = img_aug_label(cell_3_aug, 2)
cell_4_aug_label = img_aug_label(cell_4_aug, 3)
cell_5_aug_label = img_aug_label(cell_5_aug, 4)
cell_6_aug_label = img_aug_label(cell_6_aug, 5)
cell_7_aug_label = img_aug_label(cell_7_aug, 6)
cell_8_aug_label = img_aug_label(cell_8_aug, 7)
cell_9_aug_label = img_aug_label(cell_9_aug, 8)
cell_10_aug_label = img_aug_label(cell_10_aug, 9)
cell_11_aug_label = img_aug_label(cell_11_aug, 10)
cell_12_aug_label = img_aug_label(cell_12_aug, 11)

#Combine into a single (augmented) image stack
img_stack = np.vstack((cell_1_aug, cell_2_aug, cell_3_aug, cell_4_aug, cell_5_aug, cell_6_aug, 
  cell_7_aug, cell_8_aug, cell_9_aug, cell_10_aug, cell_11_aug, cell_12_aug))
img_stack_resize = img_stack.reshape(-1, 256, 256, 3) #Change to 4D
print("\nChecking the shape of the img_stack_resize: ", np.shape(img_stack_resize))

#Combine into a single label stack
cell_label_stack = np.hstack((cell_1_aug_label, cell_2_aug_label, cell_3_aug_label, cell_4_aug_label, cell_5_aug_label, 
  cell_6_aug_label, cell_7_aug_label, cell_8_aug_label, cell_9_aug_label, cell_10_aug_label, cell_11_aug_label, cell_12_aug_label))


#Shuffle both the training images and labels
def shuffle_image(img_stack_resize, cell_label_stack):
  instance = []
  shuffle_list = []
  for i in range(len(img_stack_resize)):
    img = img_stack_resize[i]
    label = cell_label_stack[i]
    instance.append(img)
    instance.append(label)
    shuffle_list.append(instance)
    instance = []

  random.shuffle(shuffle_list)

  train_img = np.empty((len(img_stack_resize), 256, 256, 3))
  train_label_raw = np.empty((len(cell_label_stack)))

  for i in range(len(img_stack_resize)):
    train_img[i] = shuffle_list[i][0]
    train_label_raw[i] = shuffle_list[i][1]

  train_label = to_categorical(train_label_raw, num_classes = no_of_class)
  print("\nChecking shape of train_label = ", np.shape(train_label))

  return train_img, train_label

train_img, train_label = shuffle_image(img_stack_resize, cell_label_stack)

#Create validation data (original unaugmented images)
val_stack = np.vstack((cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, 
  cell_7, cell_8, cell_9, cell_10, cell_11, cell_12))
val_stack_resize = val_stack.reshape(-1, 256, 256, 3)

cell_1_label = img_aug_label(cell_1, 0)
cell_2_label = img_aug_label(cell_2, 1)
cell_3_label = img_aug_label(cell_3, 2)
cell_4_label = img_aug_label(cell_4, 3)
cell_5_label = img_aug_label(cell_5, 4)
cell_6_label = img_aug_label(cell_6, 5)
cell_7_label = img_aug_label(cell_7, 6)
cell_8_label = img_aug_label(cell_8, 7)
cell_9_label = img_aug_label(cell_9, 8)
cell_10_label = img_aug_label(cell_10, 9)
cell_11_label = img_aug_label(cell_11, 10)
cell_12_label = img_aug_label(cell_12, 11)

val_label_stack = np.hstack((cell_1_label, cell_2_label, cell_3_label, cell_4_label, cell_5_label, 
  cell_6_label, cell_7_label, cell_8_label, cell_9_label, cell_10_label, cell_11_label, cell_12_label))

val_img, val_label = shuffle_image(val_stack_resize, val_label_stack)

#Final normalization
train_img = np.float16(train_img)/np.max(train_img)
val_img = np.float16(val_img)/np.max(val_img)

#np.savetxt("train_label.txt", train_label_raw)

#import model
no_of_class = 12 #6 ECM proteins x 2 cell types = 12 classes
cnn_model = seq_model()
model = cnn_model.conv_model(no_of_class)
print("\nChecking SERNN model...")
print(model.summary())

#Define model parameters
learning_rate = 0.001
decay_rate = 0.0005
model.compile(optimizer = SGD(lr = learning_rate, decay = decay_rate, momentum = 0.9, nesterov = False), 
  loss = 'categorical_crossentropy', metrics = ['accuracy'])

perf_lr_scheduler = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 50,
    verbose = 1, min_delta = 0.001, min_lr = 0.000001)

model_earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 100, restore_best_weights = True) 

history_1 = model.fit(
  #x = train_aug.flow(train_img, img_label, batch_size = 16, shuffle = True),
  train_img,
  train_label,
  batch_size = 16, #Adjust according to avoid computational resource issues
  shuffle = True, #Shuffling occurs before each epoch
  epochs = 10000, 
  validation_data = (val_img, val_label),
  validation_freq = 1,
  callbacks = [perf_lr_scheduler, model_earlystop], 
  verbose = 2)

#Save model for future uses
date = "31-MAY-2021_SERNN_1"
filename = date + '_model.h5'
model.save(filename)
filename_weight = date + "_weights.h5"
model.save_weights(filename_weight)

#Create directory if it isn't created yet
path = "Result"
try: 
  os.listdir(path)
except FileNotFoundError:
  os.mkdir(path)
else:
  print("Directory found")

#Summarize history for accuracy
def plot_training_result(history, path, index):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Accuracy & Loss')
  plt.ylabel('Accuracy/Loss')
  plt.xlabel('Epoch')
  plt.legend(['train_acc', 'val_acc', "train_loss", "val_loss"], loc = 'upper right')
  training_name = path + "/CNN_results" + str(index) + ".png"
  plt.savefig(training_name)

plot_training_result(history_1, path, 1)
