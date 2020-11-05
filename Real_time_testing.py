#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:37:40 2020

@author: pi
"""

from picamera import PiCamera
from picamera.array import PiRGBArray
import time 
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC

camera = PiCamera()

for i in range (100):
        #camera.capture("/home/pi/Desktop/newfile/" + i + ".jpg")
        rawCapture = PiRGBArray(camera)
        time.sleep(0.5)
        #t.start()
        camera.capture(rawCapture,format = 'bgr')
        I = rawCapture.array
        show = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.resize(I, (100,100))
        gray_img = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        gray_img= np.asanyarray(gray_img)
        flattened_img = gray_img.flatten()
        flattened_img = flattened_img.reshape(1, -1)
        pick = open('SVM.sav','rb')
        model = pickle.load(pick)
        pick.close()

        #Predict the new test image
        prediction = model.predict(flattened_img)
        #accuracy = model.score(xtest,ytest)

        categories = ['apple','orange']
        #print('Accuracy: ',accuracy)
        print('Prediction is:',categories[prediction[0]])
        plt.imshow(show,cmap = 'gray',vmin=0, vmax=255)
        plt.show()
#        pick_in = open('New Data1.pickle','rb')
#        data = pickle.load(pick_in)
#        pick_in.close()
#
#        random.shuffle(data)
#        features =[]
#        labels = []
# 
#        for feature ,label in data:
#            features.append(feature)
#            labels.append(label)
#
#        xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size = 0.30)
#
#
#        #model1 = SVC (C = 1,kernel = 'poly',gamma = 'auto')
#        
#        model1 = SVC (kernel = 'linear')
#        model1.fit(xtrain,ytrain)
#        prediction = model1.predict(flattened_img)
#        categories = ['apple','orange']
#        print('Prediction is:',categories[prediction [0]])
#        plt.imshow(show,cmap = 'gray')
#        plt.show()



 