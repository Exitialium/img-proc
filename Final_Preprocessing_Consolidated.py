#!/usr/bin/env python3
# -*- coding: utf-8 -*-x
"""
Created on Sun Sep 26 15:44:15 2021

@author: aryaanmehra
"""

import tensorflow as tf

from keras_preprocessing import image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


arr = []
ind_arr = []
index = 0
df = pd.read_csv('labels.csv')
for row in df["Finding Labels"]:
    #print(row)
    if '|' in row:
        arr.append(index)
        ind_arr.append(df["Image Index"][index])
    index += 1
df = df.drop(arr)
df.to_csv('labels.csv',index=False)
#print(df["Finding Labels"])


names = os.listdir('./images')
for i in names:
    if i in ind_arr:
        os.remove('./images/' + i)
#print(os.listdir('./images'))
     

for subdir, dirs, files in os.walk('./images'):
    #print(len(files))
    for file in files:
        if file != '.DS_Store' :
            newimg = Image.open('./images/'+file)
            newimg = newimg.resize((224,224))
            newimg.save('./smallimages/'+file)

'''
#only do this if you want augmented images and note that it changes the names of files
# The code only changes first 1000 images
aug_images = []

for subdir, dirs, files in os.walk('./smallimages'): 
    files = sorted(files)
    for file in files[:1001]:
        if file != '.DS_Store' :
            image = Image.open('./smallimages/' + file)
            image = np.array(image)
            if len(image.shape) == 3:
                image = np.delete(image, 2, 2)  
                image = np.delete(image, 1, 2)
                image = np.delete(image, 0, 2)
                image = image.squeeze()
            aug_images.append(image)

print(len(aug_images))
      
x_train = np.stack(aug_images, axis=0)
x_train = x_train.reshape((1000, 224, 224, 1))
print(x_train.shape)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(x_train)
save_here = './augimages'

for x, val in zip(datagen.flow(x_train,
        save_to_dir=save_here,     #this is where we figure out where to save
        save_format='png', 
        batch_size = 1000),
                  range(0)):
    pass

for subdir, dirs, files in os.walk('./smallimages'):
    #print(len(files))
    files = sorted(files)
    for file in files[1001:]:
        newimg = Image.open('./smallimages/'+file)
        newimg.save('./augimages/'+file)
'''




    
df = pd.read_csv('labels.csv')
index = 0
classes = {"Cardiomegaly":0, "No Finding":1, "Hernia":2, "Infiltration":3, "Nodule":4, "Emphysema": 5, "Effusion":6, "Atelectasis": 7, "Mass":8, "Pneumothorax":9, "Pleural_Thickening":10, "Fibrosis":11, "Consolidation":12, "Edema":13, "Pneumonia":14}
for row in df["Finding Labels"]:
    df["Finding Labels"][index] = classes[row]
    index+=1


with open('test_list.txt','r') as test:
    testlines = [t for t in test.read().splitlines()]
with open('train_val_list.txt','r') as train:
    trainlines = [t for t in train.read().splitlines()]



def get_data(filename):
    if filename in np.array(df["Image Index"]):
        data = np.expand_dims(np.asarray(Image.open('./smallimages/' + filename).convert('L')), axis=2)
        output = (data, np.array(df.loc[df['Image Index']==filename])[0][1])
        return output
    pass

arr = []
index = 0
for row in df["Image Index"]:
    if row not in trainlines:
        arr.append(index)
    index += 1
df = df.drop(arr)

counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# totals = np.array(df['Finding Labels'].value_counts())
totals = [761, 50172, 64, 7279, 2229, 578, 2759, 3383, 1690, 1232, 807, 541, 826, 397, 231]
patient_id = 0
totals = np.array(totals)
cap = totals / 3 * 2
cap = cap.astype(int)

newarr_test = []
newarr_train = []
for index, row in df.iterrows():
    i = row["Finding Labels"]
    if counts[i] == cap[i]:
        if row["Patient ID"] == patient_id:
            if get_data(row['Image Index'])!= None:
                newarr_train.append(get_data(row['Image Index']))
        else:
            if get_data(row['Image Index'])!= None:
                newarr_test.append(get_data(row['Image Index']))
    else:
        patient_id = row["Patient ID"]
        counts[i] +=1
        if get_data(row['Image Index'])!= None:
            newarr_train.append(get_data(row['Image Index']))


newarr_val = []
for file in testlines:
    if get_data(file)!= None:
        newarr_val.append(get_data(file))

newarr_val = np.array(newarr_val, dtype=object)




with open('validation.npz', 'wb') as file:
    np.savez(file, newarr_val)

with open('test.npz', 'wb') as file:
    np.savez(file, newarr_test)

with open('train.npz', 'wb') as file:
    np.savez(file, newarr_train)
    
'''
# Not sure what the point is of adding these extra steps
valarr = np.load('validation.npz', allow_pickle=True)
trainarr = np.load('train.npz', allow_pickle=True)
testarr = np.load('test.npz', allow_pickle=True)

with open('chest_train.npz','wb') as file:
    np.savez(file, arr=trainarr)

with open('chest_test.npz','wb') as file:
    np.savez(file, arr=testarr)

with open('chest_val.npz','wb') as file:
    np.savez(file, arr=valarr)
'''
#Georgia Institute of Technology


    