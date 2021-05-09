## preprocess data set 2 for the study

# install
# pip install Augmentor

# import
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import random

import os
import itertools
import glob
import shutil

import Augmentor

## check the images in the chest x ray data set
normal_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'
pneu_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'

## check images in the COVID radiography dataset
covid_path = '../input/covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/'
normal_path = '../input/covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/'
viral_path = '../input/covid19-radiography-database/COVID-19_Radiography_Dataset/Viral Pneumonia/'

## making directories
os.mkdir('/kaggle/working/train')
os.mkdir('/kaggle/working/test')
os.mkdir('/kaggle/working/valid')

os.mkdir('/kaggle/working/train/covid')
os.mkdir('/kaggle/working/test/covid')
os.mkdir('/kaggle/working/valid/covid')


os.mkdir('/kaggle/working/train/normal')
os.mkdir('/kaggle/working/test/normal')
os.mkdir('/kaggle/working/valid/normal')

os.mkdir('/kaggle/working/train/viral')
os.mkdir('/kaggle/working/test/viral')
os.mkdir('/kaggle/working/valid/viral')

os.mkdir('/kaggle/working/train/bacterial')
os.mkdir('/kaggle/working/test/bacterial')
os.mkdir('/kaggle/working/valid/bacterial')

## making directories for Augmentor 
## these files will be deleted in the end
os.mkdir('/kaggle/working/train_src')
os.mkdir('/kaggle/working/test_src')
os.mkdir('/kaggle/working/valid_src')


os.mkdir('/kaggle/working/train_src/covid')
os.mkdir('/kaggle/working/test_src/covid')
os.mkdir('/kaggle/working/valid_src/covid')

os.mkdir('/kaggle/working/train_src/normal')
os.mkdir('/kaggle/working/test_src/normal')
os.mkdir('/kaggle/working/valid_src/normal')

os.mkdir('/kaggle/working/train_src/viral')
os.mkdir('/kaggle/working/test_src/viral')
os.mkdir('/kaggle/working/valid_src/viral')

os.mkdir('/kaggle/working/train_src/bacterial')
os.mkdir('/kaggle/working/test_src/bacterial')
os.mkdir('/kaggle/working/valid_src/bacterial')

## move bacteria to one folder
os.mkdir('/kaggle/working/bacteria_merge/')

## copy model to the required directory for bacteria images
pneu_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'
len_pneu_path = len(os.listdir(pneu_path))
for img in itertools.islice(glob.iglob(os.path.join(pneu_path, '*.jpeg')), len_pneu_path):
    if img.split('/')[-1].split('_')[1] == 'bacteria':

        shutil.copy(img, '/kaggle/working/bacteria_merge/')
        
pneu_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'
len_pneu_path = len(os.listdir(pneu_path))
for img in itertools.islice(glob.iglob(os.path.join(pneu_path, '*.jpeg')), len_pneu_path):
    if img.split('/')[-1].split('_')[1] == 'bacteria':
        #print(img.split('/')[-1].split('_')[1])

        shutil.copy(img, '/kaggle/working/bacteria_merge/')
        
pneu_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'
len_pneu_path = len(os.listdir(pneu_path))
for img in itertools.islice(glob.iglob(os.path.join(pneu_path, '*.jpeg')), len_pneu_path):
    if img.split('/')[-1].split('_')[1] == 'bacteria':
        #print(img.split('/')[-1].split('_')[1])

        shutil.copy(img, '/kaggle/working/bacteria_merge/')


## make sure bacterial dataset only have one image per unique patient
## this is to prevent data leakage
patient_list = [int(i.split("_")[0][6:]) for i in os.listdir('/kaggle/working/bacteria_merge/')]
selected_list = []
for p in np.unique(patient_list):
    matched_files = [i for i in os.listdir('/kaggle/working/bacteria_merge/') if int(i.split("_")[0][6:])==p]
    if len(matched_files) ==1:
        selected_list.append('/kaggle/working/bacteria_merge/' + matched_files[0])
    else:
        #print('more')
        selected_file = random.choice(matched_files)
        #print(len(selected_file))
        selected_list.append('/kaggle/working/bacteria_merge/' + selected_file)
## delete unnecessary file
files = glob.glob('/kaggle/working/bacteria_merge/*.jpeg')
for f in files:
    #print(f)
    if f not in selected_list:
        os.remove(f)

## convert to jpg
jpgs = glob.glob('/kaggle/working/bacteria_merge/*.jpeg')

for j in jpgs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-4] + 'png', img)
## remove jpeg
files = glob.glob('/kaggle/working/bacteria_merge/*.jpeg')
for f in files:
    os.remove(f)

## set the bacterial path
bacterial_path = '/kaggle/working/bacteria_merge/'

## decide length for each set
covid_train_len = int(np.floor(len(os.listdir(covid_path))*0.5))
covid_len = len(os.listdir(covid_path))

normal_train_len = int(np.floor(len(os.listdir(normal_path))*0.5))
normal_len = len(os.listdir(normal_path))

viral_train_len = int(np.floor(len(os.listdir(viral_path))*0.5))
viral_len = len(os.listdir(viral_path))

bacterial_train_len = int(np.floor(len(os.listdir(bacterial_path))*.5))
bacterial_len = len(os.listdir(bacterial_path))
print('COVID dataset length: ', covid_len, ' Normal dataset length: ', normal_len, ' Viral dataset length: ', viral_len, bacterial dataset length', bacterial_len)

## move a certain set of image to test first
random.seed(88)

## copy image to train_src
for trainimg in itertools.islice(sorted(glob.iglob(os.path.join(viral_path, '*.png')), key=lambda k: random.random()), viral_train_len):
    shutil.copy(trainimg, '/kaggle/working/train_src/viral')

for trainimg in itertools.islice(sorted(glob.iglob(os.path.join(covid_path, '*.png')),key=lambda k: random.random()), covid_train_len):
    shutil.copy(trainimg, '/kaggle/working/train_src/covid')
    
for trainimg in itertools.islice(sorted(glob.iglob(os.path.join(normal_path, '*.png')),key=lambda k: random.random()), normal_train_len):
    shutil.copy(trainimg, '/kaggle/working/train_src/normal')

for trainimg in itertools.islice(sorted(glob.iglob(os.path.join(bacterial_path, '*.png')),key=lambda k: random.random()), bacterial_train_len):
    shutil.copy(trainimg, '/kaggle/working/train_src/bacterial/')
    
for testimg in itertools.islice(sorted(glob.iglob(os.path.join(covid_path, '*.png')),key=lambda k: random.random()), covid_train_len, covid_len):
    shutil.copy(testimg, '/kaggle/working/test_src/covid')

for testimg in itertools.islice(sorted(glob.iglob(os.path.join(normal_path, '*.png')),key=lambda k: random.random()), normal_train_len, normal_len):
    shutil.copy(testimg, '/kaggle/working/test_src/normal')

for testimg in itertools.islice(sorted(glob.iglob(os.path.join(viral_path, '*.png')), key=lambda k: random.random()),viral_train_len, viral_len):
    shutil.copy(testimg, '/kaggle/working/test_src/viral')
    
for testimg in itertools.islice(sorted(glob.iglob(os.path.join(bacterial_path, '*.png')),key=lambda k: random.random()),bacterial_train_len, bacterial_len):
    shutil.copy(testimg, '/kaggle/working/test_src/bacterial/')

## this random seed can be changed for each training
random.seed(9)
## move data from train to valid
for validimg in os.listdir('/kaggle/working/train_src/viral/'):
    if np.random.rand(1) < 0.2:
        shutil.move('/kaggle/working/train_src/viral/' + validimg, '/kaggle/working/valid_src/viral/')
        
for validimg in os.listdir('/kaggle/working/train_src/bacterial/'):
    if np.random.rand(1) < 0.2:
        shutil.move('/kaggle/working/train_src/bacterial/' + validimg, '/kaggle/working/valid_src/bacterial/')
        
for validimg in os.listdir('/kaggle/working/train_src/covid/'):
    if np.random.rand(1) < 0.2:
        shutil.move('/kaggle/working/train_src/covid/' + validimg, '/kaggle/working/valid_src/covid/')
        
for validimg in os.listdir('/kaggle/working/train_src/normal/'):
    if np.random.rand(1) < 0.2:
        shutil.move('/kaggle/working/train_src/normal/' + validimg, '/kaggle/working/valid_src/normal/')

## the result data are in train_src, valid_src 
## used for augmentor processing

