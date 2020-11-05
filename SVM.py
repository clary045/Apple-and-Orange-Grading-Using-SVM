import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap
#from matplotlib.axes._axis import _log as matplotlib_axes_logger
import pickle
import random 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

#dir = '//home//pi//Desktop//NEW_SVM'
#categories = ['NEW_Apple','NEW_Orange']
#data = []
#
#for category in categories:
#    path = os.path.join(dir,category)
#    label = categories.index(category)
#    
#    for img in os.listdir(path):
#        imgpath = os.path.join(path,img)
#        fruit_img = cv2.imread(imgpath,0)
#        
#        try:
#            fruit_img = cv2.resize(fruit_img,(100,100),cv2.IMREAD_GRAYSCALE)
#            image = np.array(fruit_img).flatten()
#            data.append([image,label])
#        except Exception as e:
#            pass
#        
#pick_in = open('New Data1.pickle','wb')
#pickle.dump(data,pick_in)
#pick_in.close()

pick_in = open('New Data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features =[]
labels = []
 
for feature ,label in data:
    features.append(feature)
    labels.append(label)

xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size = 0.30)


#model1 = SVC (C = 1,kernel = 'poly',gamma = 'auto')
model1 = SVC (C = 1, kernel = 'linear')
model1.fit(xtrain,ytrain)
#
#pick = open('SVM.sav','wb')
#model = pickle.dump(model1,pick)
#pick.close()

prediction = model1.predict(xtest)
accuracy = model1.score(xtest,ytest)
##
####categories = ['apple','orange']
print('Accuracy: ',accuracy)
###
####Making the Confusion Matrix
cm = confusion_matrix(ytest, prediction)
print('Confusion matrix',cm)
print(classification_report(ytest,prediction))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(ytest, prediction))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(ytest, prediction))

target_names = ['class 0', 'class 1']
print(classification_report(ytest, prediction, target_names=target_names))