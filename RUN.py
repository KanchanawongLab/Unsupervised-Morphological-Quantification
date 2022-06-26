import sys
import os
import glob
import random
import cv2
import hdbscan
import numpy as np
import skimage.io as skio
import time
from sklearn.manifold import TSNE
from imgaug import augmenters as iaa

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import keras
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.activations import selu
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

#Do NOT use PyQt6 with Matplotlib
import PyQt5 as pyqt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from pathlib import Path

#Let's try creating a GUI that can take in a stack of images and plot a t-SNE distribution
#Parameters ought to be editable

#Create application
class Window(QDialog):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)

        #Refer to inner classes
        self.class_1 = self.class_1(self)
        self.class_3 = self.class_3(self)

        #Create tabs
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(1000,1000)

        #Add tabs
        self.tabs.addTab(self.tab1, "Model training")
        self.tabs.addTab(self.tab2, "Cluster analysis")
        self.tabs.addTab(self.tab3, "Segmentation")

        #Create content for first tab
        #Create a grid
        self.tab1.layout = QGridLayout(self)
        self.tab2.layout = QGridLayout(self)
        self.tab3.layout = QGridLayout(self)

        #TAB 1
        #Create a train model demo button
        self.trainModelDemoButton = QPushButton("Train SERNN model (demonstration)")
        self.trainModelDemoButton.clicked.connect(self.class_1.trainModelDemo)

        #Non-demo version
        self.loadModelButton = QPushButton("Load SERNN model")
        self.loadModelButton.clicked.connect(self.class_1.loadModel)

        #Create output text box
        self.t1OutputText = QTextBrowser(self)
        self.t1OutputText.setText("Output will be shown here!")

        #Create a drop down menu to indicate number of classes to be trained (max 3)
        self.class_combo_box = QComboBox(self)
        num_class_list = ["1","2","3"]
        self.class_combo_box.addItems(num_class_list)

        #Create a import class button
        self.importClassButton  = QPushButton("Enable class import")
        self.importClassButton.clicked.connect(self.class_1.importClass)

        #Create load dataset button
        self.trainModel = QPushButton("Train SERNN model")
        self.trainModel.clicked.connect(self.class_1.trainModel)

        #Create a text box to show total number of images loaded
        self.totalImageCount = QLineEdit(self)
        self.totalImageCount.setReadOnly(True)
        self.totalImageCount_text = QLabel("Total number of images loaded:")

        #Create a button to clear/reset the dataset
        self.resetImage = QPushButton("Clear ALL images")
        self.resetImage.clicked.connect(self.class_1.resetImage)

        self.tab1.layout.addWidget(self.t1OutputText, 1, 0, 1, 4)
        self.tab1.layout.addWidget(self.trainModelDemoButton, 2, 0, 1, 1)
        self.tab1.layout.addWidget(self.loadModelButton, 3, 0, 1, 1)
        self.tab1.layout.addWidget(self.trainModel, 3, 1, 1, 1)
        self.tab1.layout.addWidget(self.class_combo_box, 4, 0, 1, 1)
        self.tab1.layout.addWidget(self.importClassButton, 4, 1, 1, 1)
        self.tab1.layout.addWidget(self.resetImage, 4, 2, 1, 1)
        self.tab1.layout.addWidget(self.totalImageCount_text, 5, 0, 1, 1)
        self.tab1.layout.addWidget(self.totalImageCount, 5, 1, 1, 3)
        
        self.tab1.setLayout(self.tab1.layout) #End of content for tab 1

        #TAB 2
        #Create a figure instance to plot on
        self.figure = plt.figure()

        #This is the Canvas Widget that displays the `figure`
        #It takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        #This is the Navigation widget
        #It takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        #Just some button connected to `plot` method
        self.plotButton = QPushButton('Plot')
        self.plotButton.clicked.connect(self.tsne_plot)
 
        #Create a quit button
        self.quitButton = QPushButton("Quit")
        self.quitButton.clicked.connect(self.closeEvent)

        #Create a load image button
        self.loadButton = QPushButton("Load images")
        self.loadButton.clicked.connect(self.importImages)

        #Create a load model button
        self.loadModelButton = QPushButton("Load model")
        self.loadModelButton.clicked.connect(self.importModel)

        #Create a HDBSCAN button
        self.hdbscanButton = QPushButton("HDBSCAN")
        self.hdbscanButton.clicked.connect(self.hdbscanPlot)

        #Create a save image (labelled) button
        self.saveButton = QPushButton("Save clustered images")
        self.saveButton.clicked.connect(self.saveImages)

        #t-SNE parameter setup text boxes
        self.ncom = QLineEdit(self)
        self.perplexity = QLineEdit(self)
        self.earlyex = QLineEdit(self)
        self.lr = QLineEdit(self)

        #Restrict to only integers and doubles
        self.onlyInt = QIntValidator()
        self.onlyDouble = QDoubleValidator()
        self.ncom.setValidator(self.onlyInt)
        self.perplexity.setValidator(self.onlyInt)
        self.earlyex.setValidator(self.onlyInt)
        self.lr.setValidator(self.onlyDouble)

        self.tsne_title = QLabel("t-SNE parameter setup")

        self.ncom_text = QLabel("Number of components:")
        self.perplexity_text = QLabel("Perplexity:")
        self.earlyex_text = QLabel("Early exaggeration:")
        self.lr_text = QLabel("Learning rate:")

        #Set default values for t-SNE parameters
        self.ncom.setText("2")
        self.perplexity.setText("4")
        self.earlyex.setText("4")
        self.lr.setText("0")

        #Create a text display box
        self.outputText = QTextBrowser(self)
        self.outputText.setText("Output/results will be shown here!")

        #Toolbar and display widgets

        #self.tab2.layout.addWidget(self.outputText,2,0,1,4)
        #self.tab2.setLayout(self.tab2.layout)

        top_row = 1
        #self.tab2.layout.addWidget(self.tabs, 2, 0, 1, 4)

        self.tab2.layout.addWidget(self.toolbar, top_row, 0, 1, 4) #row, column, no of rows, no of columns
        self.tab2.layout.addWidget(self.canvas, top_row+1, 0, 2, 4)

        #TSNE widgets
        self.tab2.layout.addWidget(self.tsne_title, top_row+3, 0)
        self.tab2.layout.addWidget(self.ncom_text, top_row+4, 0, 1, 1)
        self.tab2.layout.addWidget(self.ncom, top_row+4, 1, 1, 1)
        self.tab2.layout.addWidget(self.perplexity_text, top_row+4, 2, 1, 1)
        self.tab2.layout.addWidget(self.perplexity, top_row+4, 3, 1, 1)
        self.tab2.layout.addWidget(self.earlyex_text, top_row+5, 0, 1, 1)
        self.tab2.layout.addWidget(self.earlyex, top_row+5, 1, 1, 1)
        self.tab2.layout.addWidget(self.lr_text, top_row+5, 2, 1, 1)
        self.tab2.layout.addWidget(self.lr, top_row+5, 3, 1, 1)

        #Output box
        self.tab2.layout.addWidget(self.outputText, top_row+6, 0, 1, 4)

        #Rest of the buttons
        self.tab2.layout.addWidget(self.loadButton, top_row+7, 0, 1, 1)
        self.tab2.layout.addWidget(self.loadModelButton, top_row+7, 1, 1, 1)
        self.tab2.layout.addWidget(self.plotButton, top_row+8, 0, 1, 1)
        self.tab2.layout.addWidget(self.hdbscanButton, top_row+8, 1, 1, 1)
        self.tab2.layout.addWidget(self.saveButton, top_row+9, 0, 1, 1)
        self.tab2.layout.addWidget(self.quitButton, top_row+9, 3, 2, 1)

        self.tab2.setLayout(self.tab2.layout) #End of content for tab 2

        #TAB 3

        #Create output text box
        self.t3OutputText = QTextBrowser(self)
        self.t3OutputText.setText("Output will be shown here!")

        #Load segmentation model
        self.loadSegModelButton = QPushButton("Load SERNN model")
        self.loadSegModelButton.clicked.connect(self.class_3.loadModel)

        #Create buttons and text lines for importing of images and masks
        self.importImageButton = QPushButton("Load images")
        self.importImageButton.clicked.connect(self.class_3.importImage)
        self.image_dir = QLineEdit(self)
        self.image_dir.setReadOnly(True)
        self.importMaskButton = QPushButton("Load masks")
        self.importMaskButton.clicked.connect(self.class_3.importMask)
        self.mask_dir = QLineEdit(self)
        self.mask_dir.setReadOnly(True)

        #Train model button
        self.trainSegModel = QPushButton("Train model")
        self.trainSegModel.clicked.connect(self.class_3.trainModel)

        #Load trained model file
        self.loadTrainedSegModel = QPushButton("Load trained segmentation model")
        self.loadTrainedSegModel.clicked.connect(self.class_3.loadTrainedModel)

        #Create buttons for loading and segmentation of actin, mt and nucleus components
        self.loadActinButton = QPushButton("Load and segment actin files")
        self.loadActinButton.clicked.connect(self.class_3.segmentActin)
        self.loadMicroButton = QPushButton("Load and segment microtubule files")
        self.loadMicroButton.clicked.connect(self.class_3.segmentMicro)
        self.loadNucleusButton = QPushButton("Load and segment nucleus files")
        self.loadNucleusButton.clicked.connect(self.class_3.segmentNucleus)
        self.actin_dir = QLineEdit(self)
        self.actin_dir.setReadOnly(True)
        self.micro_dir = QLineEdit(self)
        self.micro_dir.setReadOnly(True)
        self.nucleus_dir = QLineEdit(self)
        self.nucleus_dir.setReadOnly(True)

        self.tab3.layout.addWidget(self.t3OutputText, 1, 0, 1, 4)
        self.tab3.layout.addWidget(self.loadSegModelButton, 2, 0, 1, 1)
        self.tab3.layout.addWidget(self.trainSegModel, 2, 1, 1, 1)
        
        self.tab3.layout.addWidget(self.importImageButton, 3, 0, 1, 1)
        self.tab3.layout.addWidget(self.image_dir, 3, 1, 1, 3)
        self.tab3.layout.addWidget(self.importMaskButton, 4, 0, 1, 1)
        self.tab3.layout.addWidget(self.mask_dir, 4, 1, 1, 3)

        self.tab3.layout.addWidget(self.loadTrainedSegModel, 5, 0, 1, 1)
        self.tab3.layout.addWidget(self.loadActinButton, 6, 0, 1, 1)
        self.tab3.layout.addWidget(self.actin_dir, 6, 1, 1, 3)
        self.tab3.layout.addWidget(self.loadMicroButton, 7, 0, 1, 1)
        self.tab3.layout.addWidget(self.micro_dir, 7, 1, 1, 3)
        self.tab3.layout.addWidget(self.loadNucleusButton, 8, 0, 1, 1)
        self.tab3.layout.addWidget(self.nucleus_dir, 8, 1, 1, 3)



        

        #FRANGI FOR MICROTUBULES?????????????????????????????????????????





        self.tab3.setLayout(self.tab3.layout) #End of content for tab 3

        #Add tabs to the GUI
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        #Add a QMainWindow to add menu bars and so on
        #mainWindow = QMainWindow()

        #self.statusbar = mainWindow.statusBar()
        #grid.addWidget(self.statusbar, top_row+10, 0)
        #self.statusbar.showMessage("Status can be seen here!")
        #self.setMouseTracking(True) #Displays status when mouse is hovering over an item

        #menubar = mainWindow.menuBar()
        #grid.addWidget(menubar, 2, 0, 1, 4)

        #First menu category
        #self.fileMenu = menubar.addMenu('&File')

        #self.importAct = QAction("Import images", self)
        #self.importAct.triggered.connect(self.importImages)
        #self.importAct.setStatusTip("Import your images")
        #self.fileMenu.addAction(self.importAct)

    #TAB 1 FUNCTIONS
    class class_1():
        #Accessing parent from child class
        #https://stackoverflow.com/questions/69138209/how-to-access-parent-class-instance-attribute-from-child-class-instance
        def __init__(self, parent):
            self.window = parent
            self.customDataset = [] #We will load images into this function
            self.customLabel = [] #We will do the labels automatically
            self.numClass = 0 #Determines the number of classes for training

        def trainModelDemo(self):
            import GUI_001_train_SERNN_model #We will train the model using this script
            #This is purely for demonstration purposes
            #Users should use the other functions in tab 1 if they want to experiment with other models and datasets

        def loadModel(self):
            #Edit the model by editing the script directly
            #No idea how to import python files without hard coding the directory
            from SERNN_model import seq_model
            cnn_model = seq_model()
            self.untrainedModel = cnn_model.conv_model(self.numClass)
            toText = []
            self.untrainedModel.summary(print_fn = lambda x: toText.append(x))
            model_summary = "\n".join(toText)
            self.window.t1OutputText.setText("Model loaded! \n\n{}".format(model_summary))

        def trainModel(self):
            len_dataset = len(self.customDataset)
            len_aug = len_dataset*15
            self.window.t1OutputText.setText(str(self.customLabel))
            self.augData = np.empty((len_aug, 256, 256, 3))
            self.augLabel = np.empty((len_aug,))
            self.window.t1OutputText.setText("Performing augmentation now...")
            for i in range(len(self.customDataset)):
                img = self.customDataset[i]
                img = np.float32(img)/np.max(img)
                img = np.uint8(img*255)
                img_4d = img.reshape(-1, 256, 256, 3)
                label = self.customLabel[i]
                aug_img = self.img_aug(img_4d)
                if i == 0:
                    for j in range(len(aug_img)):
                        self.augData[j] = aug_img[j]
                        self.augLabel[j] = label
                elif i > 0:
                    for j in range(len(aug_img)):
                        self.augData[j + (i*15)] = aug_img[j]
                        self.augLabel[j + (i*15)] = label

            self.augData = np.float32(self.augData)/np.max(self.augData)
            self.trainData, self.trainLabel = self.shuffle_image(self.augData, self.augLabel)
            self.window.t1OutputText.setText("Total number of images and labels: {}, {}\nNumber of classes: {}"
                .format(len(self.trainData), len(self.trainLabel), self.numClass))
            time.sleep(10)

            #Validation data
            self.valData, self.valLabel = self.shuffle_image(self.customDataset, self.customLabel)
            self.valData = np.float32(self.valData)/np.max(self.valData)

            learning_rate = 0.001
            decay_rate = 0.0005

            self.untrainedModel.compile(optimizer = SGD(lr = learning_rate, decay = decay_rate, momentum = 0.9, nesterov = False), 
                loss = 'categorical_crossentropy', metrics = ['accuracy'])

            perf_lr_scheduler = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 20, verbose = 1, min_delta = 0.001, min_lr = 0.000001)

            model_earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 50, restore_best_weights = True) 

            model_result = self.untrainedModel.fit(
                self.trainData,
                self.trainLabel,
                batch_size = 16,
                shuffle = True,
                epochs = 10000, 
                validation_data = (self.valData, self.valLabel),
                validation_freq = 1,
                callbacks = [perf_lr_scheduler, model_earlystop], 
                verbose = 2)

            model_name = "Trained_SERNN_model.h5"
            self.untrainedModel.save(model_name)
            weight_name = "Trained_SERNN_weights.h5"
            self.untrainedModel.save_weights(weight_name)

            self.plot_training_result(model_result, 1)

        def resetImage(self):
            self.customDataset = []
            self.customLabel = []
            self.window.t1OutputText.setText("All images have been cleared!")
            self.window.totalImageCount.setText(str(len(self.customDataset)))

        def importClass(self):
            self.num_class = int(self.window.class_combo_box.currentText())
            if self.num_class == 1:
                self.window.t1OutputText.setText("You can now import {} class!".format(self.num_class))
            elif self.num_class > 1:
                self.window.t1OutputText.setText("You can now import {} classes!".format(self.num_class))

            #We create import functions according to the number of class selected
            button_row = 6

            if self.num_class == 1:
                self.importClass_1 = QPushButton("Load class 1")
                self.importClass_1.clicked.connect(self.importClassImage_1)
                self.class_1_dir = QLineEdit(self.window)
                self.class_1_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_1, button_row, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_1_dir, button_row, 1, 1, 3)

                self.numClass = 1

                #self.window.tab1.layout.removeWidget(self.importClass_2)
                #self.window.tab1.layout.removeWidget(self.class_2_dir)
                #self.window.tab1.layout.removeWidget(self.importClass_3)
                #self.window.tab1.layout.removeWidget(self.class_3_dir)

            elif self.num_class == 2:
                self.importClass_1 = QPushButton("Load class 1")
                self.importClass_1.clicked.connect(self.importClassImage_1)
                self.class_1_dir = QLineEdit(self.window)
                self.class_1_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_1, button_row, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_1_dir, button_row, 1, 1, 3)

                self.importClass_2 = QPushButton("Load class 2")
                self.importClass_2.clicked.connect(self.importClassImage_2)
                self.class_2_dir = QLineEdit(self.window)
                self.class_2_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_2, button_row+1, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_2_dir, button_row+1, 1, 1, 3)

                self.numClass = 2

                #self.window.tab1.layout.removeWidget(self.importClass_3)
                #self.window.tab1.layout.removeWidget(self.class_3_dir)

            elif self.num_class == 3:
                self.importClass_1 = QPushButton("Load class 1")
                self.importClass_1.clicked.connect(self.importClassImage_1)
                self.class_1_dir = QLineEdit(self.window)
                self.class_1_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_1, button_row, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_1_dir, button_row, 1, 1, 3)

                self.importClass_2 = QPushButton("Load class 2")
                self.importClass_2.clicked.connect(self.importClassImage_2)
                self.class_2_dir = QLineEdit(self.window)
                self.class_2_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_2, button_row+1, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_2_dir, button_row+1, 1, 1, 3)

                self.importClass_3 = QPushButton("Load class 3")
                self.importClass_3.clicked.connect(self.importClassImage_3)
                self.class_3_dir = QLineEdit(self.window)
                self.class_3_dir.setReadOnly(True)
                self.window.tab1.layout.addWidget(self.importClass_3, button_row+2, 0, 1, 1)
                self.window.tab1.layout.addWidget(self.class_3_dir, button_row+2, 1, 1, 3)

                self.numClass = 3
            
        def importClassImage_1(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                dataset = self.load_image(folder)
            except FileNotFoundError:
                return None
            for i in range(len(dataset)):
                self.customDataset.append(dataset[i])
                self.customLabel.append(int("0"))
            showText = "Number of class 1 images loaded: {}\nTotal number of images loaded: {}".format(len(dataset), len(self.customDataset))
            self.window.t1OutputText.setText(showText)
            self.class_1_dir.setText(folder)
            self.window.totalImageCount.setText(str(len(self.customDataset)))

        def importClassImage_2(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                dataset = self.load_image(folder)
            except FileNotFoundError:
                return None
            for i in range(len(dataset)):
                self.customDataset.append(dataset[i])
                self.customLabel.append(int("1"))
            showText = "Number of class 2 images loaded: {}\nTotal number of images loaded: {}".format(len(dataset), len(self.customDataset))
            self.window.t1OutputText.setText(showText)
            self.class_2_dir.setText(folder)
            self.window.totalImageCount.setText(str(len(self.customDataset)))

        def importClassImage_3(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                dataset = self.load_image(folder)
            except FileNotFoundError:
                return None
            for i in range(len(dataset)):
                self.customDataset.append(dataset[i])
                self.customLabel.append(int("2"))
            showText = "Number of class 3 images loaded: {}\nTotal number of images loaded: {}".format(len(dataset), len(self.customDataset))
            self.window.t1OutputText.setText(showText)
            self.class_3_dir.setText(folder)
            self.window.totalImageCount.setText(str(len(self.customDataset)))

        def load_image(self, image_dir):
            image_list = []
            self.num_img = 0
            print("\nLoading images now...")
            for filename in os.listdir(image_dir):
                img = cv2.imread(os.path.join(image_dir, filename))
                if img is not None:
                    img = np.float16(img)/np.max(img)
                    img = np.uint8(img*255)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.float16(img)/np.max(img)
                    image_list.append(img)
                    self.num_img += 1
          
            image_np = np.empty((self.num_img, 256, 256, 3), dtype = np.float32)
            print("\nNumber of images loaded: {}".format(self.num_img))

            for i in range(self.num_img):
                image_np[i,:,:,:] = image_list[i]

            return image_np

        #Perform initial augmentation via iaa
        def img_aug(self, img_stack):
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

            #Perform augmentations
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
            #print("Memory size of array: {} GB".format(float(cell_stack.nbytes/1000000000)))

            return cell_stack

        def shuffle_image(self, img_stack_resize, cell_label_stack):
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

            print(np.shape(train_img))
            print(np.shape(train_label_raw))
            print(train_label_raw)
            print(self.numClass)

            train_label = to_categorical(train_label_raw, num_classes = self.numClass)
                
            return train_img, train_label

        def plot_training_result(self, history, index):
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Accuracy & Loss')
            plt.ylabel('Accuracy/Loss')
            plt.xlabel('Epoch')
            plt.legend(['train_acc', 'val_acc', "train_loss", "val_loss"], loc = 'upper right')
            training_name = "SERNN_result_" + str(index) + ".png"
            plt.savefig(training_name)

    #TAB 2 FUNCTIONS

    def importModel(self):
        filename = QFileDialog.getOpenFileName(self, "Select model file", "./")
        modelFile = filename[0]
        try:
            self.model = load_model(modelFile)
        except OSError:
            self.outputText.setText("ERROR - Please upload only .h5 or .h5py files!")
            return
        else:
            toText = []
            self.model.summary(print_fn = lambda x: toText.append(x))
            model_summary = "\n".join(toText)
            self.outputText.setText("Model loaded! \n\n{}".format(model_summary))

    def tsne_plot(self):
        #Perform t-SNE dimensionality reduction on dataset
        try:
            data_vector = self.dataset.reshape(-1, 256*256*3)
        except AttributeError:
            self.outputText.setText("ERROR - No images loaded!")
            return

        #Load images into SERNN model
        self.outputText.setText("Extracting morphological features using SERNN model now...")
        output_vector = np.empty((len(self.dataset), 128))
        layer_output = [layer.output for layer in self.model.layers[:]]
        cluster_model = Model(inputs = self.model.input, outputs = layer_output)

        for i in range(len(self.dataset)):
            img = self.dataset[i]
            img_4D = img.reshape(-1, 256, 256, 3)
            all_layers = cluster_model.predict(img_4D)
            output_vector[i] = all_layers[-3]

        #Set default values
        if self.lr.text() == '0':
            learning_rate = str(float(len(output_vector)/12))
            self.lr.setText(learning_rate)

        #Override default values
        ncom = int(self.ncom.text())
        perplexity = int(self.perplexity.text())
        earlyex = int(self.earlyex.text())
        lr = float(self.lr.text())

        showText = "Performing t-SNE dimensionality reduction now..."
        self.outputText.setText(showText)

        reduced_data = TSNE(n_components = ncom, perplexity = perplexity, early_exaggeration = 4, 
            learning_rate = lr, random_state = 42, init = "pca").fit_transform(output_vector)
        self.tsne_data = reduced_data.copy() #Make a copy for other uses

        #Instead of ax.hold(False)
        self.figure.clear()

        #Create an axis
        ax = self.figure.add_subplot(111)

        #Plot data
        reduced_x = []
        reduced_y = []

        for j in range(len(reduced_data)):
            reduced_x.append(reduced_data[j][0])
            reduced_y.append(reduced_data[j][1])

        ax.scatter(reduced_x, reduced_y, facecolor = 'red')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel('t-SNE 2')

        #Refresh canvas
        self.canvas.draw()

        #Display text
        showText = "t-SNE scatter plot displayed!"
        self.outputText.setText(showText)

    def hdbscanPlot(self):
        try:
            data_length = len(self.tsne_data)
        except AttributeError:
            self.outputText.setText("ERROR - no t-SNE data available!")
            return

        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size = 5)

        self.cluster_data = hdbscan_model.fit_predict(self.tsne_data)
        print(self.cluster_data)

        #We will determine the number of outliers and clusters here
        outlier_num = 0
        cluster_list = []
        cluster_num = 0
        for i in range(len(self.cluster_data)):
            dup_num = False
            if str(self.cluster_data[i]) == '-1':
                outlier_num += 1
            elif str(self.cluster_data[i]) != '-1':
                if len(cluster_list) == 0:
                    cluster_list.append(self.cluster_data[i])
                    cluster_num += 1
                if len(cluster_list) > 0:
                    for j in range(len(cluster_list)):
                        if cluster_list[j] == self.cluster_data[i]:
                            dup_num = True
                    if dup_num == False:
                        cluster_list.append(self.cluster_data[i])
                        cluster_num += 1

        #Instead of ax.hold(False)
        self.figure.clear()

        #Create an axis
        ax = self.figure.add_subplot(111)

        #Plot data
        reduced_x = []
        reduced_y = []

        for j in range(len(self.tsne_data)):
            reduced_x.append(self.tsne_data[j][0])
            reduced_y.append(self.tsne_data[j][1])

        ax.scatter(reduced_x, reduced_y, c = self.cluster_data)
        ax.set_xlabel("HDBSCAN 1")
        ax.set_ylabel('HDBSCAN 2')

        #Refresh canvas
        self.canvas.draw()

        self.outputText.setText("HDBSCAN plot done! \n\nNumber of clusters: {} \nNumber of outliers: {}".format(cluster_num, outlier_num))

    def saveImages(self):
        try:
            cluster_length = len(self.cluster_data)
        except AttributeError:
            self.outputText.setText("ERROR - no cluster data available!")
            return

        #Create a folder in the same directory
        path = "HDBSCAN_cluster/"
        try:
            os.listdir(path)
        except FileNotFoundError:
            os.mkdir(path)

        for i in range(len(self.dataset)):
            img = self.dataset[i]
            label = str(int(self.cluster_data[i]))

            #Convert image to uint8 type first for easy preview purposes
            img = np.float32(img)/np.max(img)
            img = np.uint8(img*255)

            name = path + "cluster_" + label + "-" + str(i) + ".tif"

            skio.imsave(name, img)

        self.outputText.setText("Images have been saved with the cluster labels!")

    def importImages(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select directory"))
        self.showImagesLoaded(folder)

    def load_image(self, image_dir):
        image_list = []
        self.num_img = 0
        print("\nLoading images now...")
        for filename in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir, filename))
            if img is not None:
                img = np.float16(img)/np.max(img)
                img = np.uint8(img*255)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.float16(img)/np.max(img)
                image_list.append(img)
                self.num_img += 1
      
        image_np = np.empty((self.num_img, 256, 256, 3), dtype = np.float16)
        print("\nNumber of images loaded: {}".format(self.num_img))

        for i in range(self.num_img):
            image_np[i,:,:,:] = image_list[i]

        return image_np

    def showImagesLoaded(self, image_dir):
        self.dataset = self.load_image(image_dir)
        try: 
            self.dataset = self.load_image(image_dir)
        except FileNotFoundError:
            return None
        showText = "Number of images loaded: " + str(self.num_img)
        self.outputText.setText(showText)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Do you want to quit?",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            #event.accept()
            sys.exit()
        else:
            #event.ignore()
            pass

    class class_3():
        #Accessing parent from child class
        #https://stackoverflow.com/questions/69138209/how-to-access-parent-class-instance-attribute-from-child-class-instance
        def __init__(self, parent):
            self.window = parent

        def loadModel(self):
            #Edit the model by editing the script directly
            #No idea how to import python files without hard coding the directory
            from SERNN_segmentation_model import seq_model
            cnn_model = seq_model()
            self.model = cnn_model.conv_model()
            toText = []
            self.model.summary(print_fn = lambda x: toText.append(x))
            model_summary = "\n".join(toText)
            self.window.t3OutputText.setText("Model loaded! \n\n{}".format(model_summary))

        def importImage(self):
            showText = "WARNING: Please make sure your images and masks are in the exact same order!"
            self.window.t3OutputText.setText(showText)
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                self.imageList = self.load_image(folder)
            except FileNotFoundError:
                return None
            showText = "Number of images loaded: {}".format(len(self.imageList))
            self.window.t3OutputText.setText(showText)
            self.window.image_dir.setText(folder)
            self.window.totalImageCount.setText(str(len(self.imageList)))

        def importMask(self):
            showText = "WARNING: Please make sure your images and masks are in the exact same order!"
            self.window.t3OutputText.setText(showText)
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                self.maskList = self.load_image(folder)
            except FileNotFoundError:
                return None
            showText = "Number of masks loaded: {}".format(len(self.maskList))
            self.window.t3OutputText.setText(showText)
            self.window.mask_dir.setText(folder)
            self.window.totalImageCount.setText(str(len(self.maskList)))

        def trainModel(self):
            len_dataset = len(self.imageList)
            len_aug = len_dataset*13
            self.window.t3OutputText.setText("Performing augmentation now...")
            self.augImage = self.im_aug(self.imageList)
            self.augMask = self.im_aug(self.maskList)

            self.augImage_shuffle, self.augMask_shuffle = self.shuffle_image(self.augImage, self.augMask)
            self.augImage_shuffle = np.float32(self.augImage_shuffle)/np.max(augImage_shuffle)
            self.augMask_shuffle = np.float32(self.augMask_shuffle)/np.max(augMask_shuffle)
            split = int(0.7 * len(train_resize))
            self.trainImage = self.augImage_shuffle[:split]
            self.trainMask = self.augMask_shuffle[:split]
            self.valImage = self.augImage_shuffle[split:]
            self.valMask = self.augMask_shuffle[split:]
        
            #Validation data
            self.valData, self.valLabel = self.shuffle_image(self.customDataset, self.customLabel)
            self.valData = np.float32(self.valData)/np.max(self.valData)

            learning_rate = 0.001
            decay_rate = 0.0005

            self.model.compile(optimizer = SGD(lr = learning_rate, decay = decay_rate, momentum = 0.9, nesterov = False), 
                loss = 'binary_crossentropy', metrics = ['accuracy'])

            perf_lr_scheduler = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 20, verbose = 1, min_delta = 0.001, min_lr = 0.000001)

            model_earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 50, restore_best_weights = True) 

            model_result = self.model.fit(
                self.trainImage,
                self.trainMask,
                batch_size = 16,
                shuffle = True,
                epochs = 10000, 
                validation_data = (self.valImage, self.valMask),
                validation_freq = 1,
                callbacks = [perf_lr_scheduler, model_earlystop], 
                verbose = 2)

            model_name = "Trained_SERNN_segmentation_model.h5"
            self.model.save(model_name)
            weight_name = "Trained_SERNN__segmentation_weights.h5"
            self.model.save_weights(weight_name)

        def loadTrainedModel(self):
            filename = QFileDialog.getOpenFileName(self.window, "Select model file", "./")
            modelFile = filename[0]
            try:
                self.segModel = load_model(modelFile)
            except OSError:
                self.window.t3OutputText.setText("ERROR - Please upload only .h5 or .h5py files!")
                return
            else:
                toText = []
                self.segModel.summary(print_fn = lambda x: toText.append(x))
                model_summary = "\n".join(toText)
                self.window.t3OutputText.setText("Model loaded! \n\n{}".format(model_summary))

        def segmentActin(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                self.actinList = self.load_image(folder)
            except FileNotFoundError:
                return None

            #Create output list   
            self.actinOutput = np.empty((len(self.actinList), 256, 256))

            for img in self.actinList:
                if img is not None:
                    img = np.float32(img)/np.max(img)
                    img = np.uint8(img*255)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.float32(img)/np.max(img)
                    input_img = np.empty((1, 256, 256, 1), dtype = np.float32)
                    input_img[0,:,:,0] = img
                    output = self.segModel.predict(input_img)
                    output = np.float32(output)/np.max(output)
                    output = np.uint8(output*255)

                    self.actinOutput[i,:,:] = output[i]

            self.saveImages(self.actinOutput)

        def segmentMicro(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                self.microList = self.load_image(folder)
            except FileNotFoundError:
                return None

            #Create output list   
            self.microOutput = np.empty((len(self.microList), 256, 256))

            for img in self.microList:
                if img is not None:
                    img = np.float32(img)/np.max(img)
                    img = np.uint8(img*255)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.float32(img)/np.max(img)
                    input_img = np.empty((1, 256, 256, 1), dtype = np.float32)
                    input_img[0,:,:,0] = img
                    output = self.segModel.predict(input_img)
                    output = np.float32(output)/np.max(output)
                    output = np.uint8(output*255)

                    self.microOutput[i,:,:] = output[i]

            self.saveImages(self.microOutput)

        def segmentNucleus(self):
            folder = str(QFileDialog.getExistingDirectory(self.window, "Select directory"))
            try: 
                self.nucleusList = self.load_image(folder)
            except FileNotFoundError:
                return None

            #Create output list   
            self.nucleusOutput = np.empty((len(self.nucleusList), 256, 256))

            for img in self.actinList:
                if img is not None:
                    img = np.float32(img)/np.max(img)
                    img = np.uint8(img*255)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.float32(img)/np.max(img)
                    input_img = np.empty((1, 256, 256, 1), dtype = np.float32)
                    input_img[0,:,:,0] = img
                    output = self.segModel.predict(input_img)
                    output = np.float32(output)/np.max(output)
                    output = np.uint8(output*255)

                    self.nucleusOutput[i,:,:] = output[i]

            self.saveImages(self.nucleusOutput)

        def load_image(self, image_dir):
            image_list = []
            self.num_img = 0
            print("\nLoading images now...")
            for filename in os.listdir(image_dir):
                img = cv2.imread(os.path.join(image_dir, filename))
                if img is not None:
                    img = np.float16(img)/np.max(img)
                    img = np.uint8(img*255)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.float16(img)/np.max(img)
                    image_list.append(img)
                    self.num_img += 1
          
            image_np = np.empty((self.num_img, 256, 256, 1), dtype = np.float32)
            print("\nNumber of images loaded: {}".format(self.num_img))

            for i in range(self.num_img):
                image_np[i,:,:,0] = image_list[i]

            return image_np

        def saveImages(self, output_list):
            try:
                cluster_length = len(output_list)
            except AttributeError:
                self.window.t3OutputText.setText("ERROR - no segmentation data available!")
                return

            #Create a folder in the same directory
            if self.actinOutput == output_list:
                path = "actin_seg/"
            if self.microOutput == output_list:
                path = "micro_seg/"
            if self.nucleusOutput == output_list:
                path = "nucleus_seg/"

            try:
                os.listdir(path)
            except FileNotFoundError:
                os.mkdir(path)

            index_1 = 0
            index_2 = 0
            index_3 = 0

            for i in range(len(output_list)):
                img = output_list[i]

                #Convert image to uint8 type first for easy preview purposes
                img = np.float32(img)/np.max(img)
                img = np.uint8(img*255)

                name = path + "img_" + str(index_1) + str(index_2) + str(index_3) + ".tif"

                skio.imsave(name, img)

                index_3 += 1
                if index_3 == 10:
                    index_2 += 1
                    index_3 = 0
                if index_2 == 10:
                    index_1 += 1
                    index_2 = 0

            self.window.t3OutputText.setText("Segmented images have been saved!")

        def img_aug(self, img_stack, mask = False):
            flip_lr = iaa.Sequential([iaa.Fliplr(1.0)])
            flip_ud = iaa.Sequential([iaa.Flipud(1.0)])
            rotate_45 = iaa.Sequential([iaa.Affine(rotate = 45)])
            rotate_90 = iaa.Sequential([iaa.Affine(rotate = 90)])
            rotate_135 = iaa.Sequential([iaa.Affine(rotate = 135)])
            rotate_180 = iaa.Sequential([iaa.Affine(rotate = 180)])
            rotate_225 = iaa.Sequential([iaa.Affine(rotate = 225)])
            rotate_270 = iaa.Sequential([iaa.Affine(rotate = 270)])
            rotate_315 = iaa.Sequential([iaa.Affine(rotate = 315)])
            gaussian_noise = iaa.Sequential([iaa.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.05*255))])
            multiply = iaa.Sequential([iaa.Multiply((0.8, 1.2))])
            dropout = iaa.Sequential([iaa.Dropout(p = (0, 0.3))])
            salt_pepper = iaa.Sequential([iaa.SaltAndPepper(0.1)])

            #Create empty arrays
            cell_lr = np.empty((len(img_stack), 256, 256, 1))
            cell_ud = np.empty((len(img_stack), 256, 256, 1))
            cell_45 = np.empty((len(img_stack), 256, 256, 1))
            cell_90 = np.empty((len(img_stack), 256, 256, 1))
            cell_135 = np.empty((len(img_stack), 256, 256, 1))
            cell_180 = np.empty((len(img_stack), 256, 256, 1))
            cell_225 = np.empty((len(img_stack), 256, 256, 1))
            cell_270 = np.empty((len(img_stack), 256, 256, 1))
            cell_315 = np.empty((len(img_stack), 256, 256, 1))
            cell_noise = np.empty((len(img_stack), 256, 256, 1))
            cell_multiply = np.empty((len(img_stack), 256, 256, 1))
            cell_saltpepper = np.empty((len(img_stack), 256, 256, 1))

            #Perform augmentations
            for i in range(len(img_stack)):
                cell_lr[i] = flip_lr(image = img_stack[i])

            for i in range(len(img_stack)):
                cell_ud[i] = flip_ud(image = img_stack[i])

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

            if mask == False:
                cell_stack = np.vstack((cell_lr, cell_ud, cell_45, cell_90, cell_135, cell_180, cell_225, cell_270, cell_315, cell_noise, cell_multiply, cell_saltpepper))
            elif mask == True:
                cell_stack = np.vstack((cell_lr, cell_ud, cell_45, cell_90, cell_135, cell_180, cell_225, cell_270, cell_315, img_stack, img_stack, img_stack))

            return cell_stack

        def shuffle_image(self, img_stack, mask_stack):
            instance = []
            shuffle_list = []
            for i in range(len(img_stack)):
                img = img_stack[i]
                mask = mask_stack[i]
                instance.append(img)
                instance.append(mask)
                shuffle_list.append(instance)
                instance = []

            random.shuffle(shuffle_list)

            train_img = np.empty((len(img_stack), 256, 256, 1))
            train_mask = np.empty((len(mask_stack), 256, 256, 1))

            for i in range(len(img_stack)):
                train_img[i] = shuffle_list[i][0]
                train_mask[i] = shuffle_list[i][1]

            return train_img, train_mask


def main():
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec()) #Ensures a clean exit

if __name__ == '__main__':
    main()