## from processed data to model results
## this excldues preprossessing_data_set2.py

## imports
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
from sklearn.metrics import precision_recall_curve


## augmentor
## Process covid directory first
p = Augmentor.Pipeline()
p.add_further_directory(new_source_directory='/kaggle/working/train_src/covid/', new_output_directory='/kaggle/working/train/covid/')
p.resize(width=244, height=244, probability=1)
p.crop_centre(probability=.1, percentage_area=.9)
p.resize(width=244, height=244, probability=1)
p.sample(3000)
## test dir 
p2 = Augmentor.Pipeline()
p2.add_further_directory(new_source_directory='/kaggle/working/test_src/covid/', new_output_directory='/kaggle/working/test/covid/')
p2.resize(width=244, height=244, probability=1)
p2.crop_centre(probability=.1, percentage_area=.9)
p2.resize(width=244, height=244, probability=1)

p2.process()

## valid dir
p3 = Augmentor.Pipeline()
p3.add_further_directory(new_source_directory='/kaggle/working/valid_src/covid/', new_output_directory='/kaggle/working/valid/covid/')
p3.resize(width=244, height=244, probability=1)
p3.crop_centre(probability=.1, percentage_area=.9)
p3.resize(width=244, height=244, probability=1)

p3.process()

## other directory are the same
for category in ['bacterial', 'viral', 'normal']:
    ## train
    p = Augmentor.Pipeline()
    p.add_further_directory(new_source_directory='/kaggle/working/train_src/' + category + '/', new_output_directory='/kaggle/working/train/' + category + '/')
    p.resize(width=244, height=244, probability=1)
    p.crop_centre(probability=.1, percentage_area=.9)
    p.resize(width=244, height=244, probability=1)

    p.process()
    
    ## test
    p2 = Augmentor.Pipeline()
    p2.add_further_directory(new_source_directory='/kaggle/working/test_src/' + category + '/', new_output_directory='/kaggle/working/test/' + \
                             category + '/')
    p2.resize(width=244, height=244, probability=1)
    p2.crop_centre(probability=.1, percentage_area=.9)
    p2.resize(width=244, height=244, probability=1)
    
    p2.process()
    
    ## valid
    p3 = Augmentor.Pipeline()
    p3.add_further_directory(new_source_directory='/kaggle/working/valid_src/' + category + '/', new_output_directory='/kaggle/working/valid/' + category+ '/')
    p3.resize(width=244, height=244, probability=1)
    p3.crop_centre(probability=.1, percentage_area=.9)
    p3.resize(width=244, height=244, probability=1)
    
    p3.process()

## remove all the train_src dir

files = glob.glob('/kaggle/working/train_src/*/*.png')
for f in files:
    os.remove(f)
files = glob.glob('/kaggle/working/test_src/*/*.png')
for f in files:
    os.remove(f)
files = glob.glob('/kaggle/working/valid_src/*/*.png')
for f in files:
    os.remove(f)
files = glob.glob('/kaggle/working/bacteria_merge/*.png')
for f in files:
    os.remove(f)

## normalize
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ]),
    'validation': transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
}

## data loader
data_images = {
    'train': datasets.ImageFolder('/kaggle/working/train', data_transforms['train']),
    'validation': datasets.ImageFolder('/kaggle/working/valid', data_transforms['validation'])
}
dataloaders = {
    'train': torch.utils.data.DataLoader(data_images['train'], batch_size=32, shuffle=True, num_workers=0),
    'validation': torch.utils.data.DataLoader(data_images['validation'], batch_size=32,shuffle=False,num_workers=0)
}

testloaders = {
    'test': torch.utils.data.DataLoader(datasets.ImageFolder('/kaggle/working/test/', data_transforms['validation']), \
                                       batch_size=32, shuffle=False, num_workers=0)
}


## freeze the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True).cuda()

for param in model.parameters():
    param.requires_grad = False

## replace last fc layer
model.fc = nn.Sequential(
    nn.Linear(2048, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 4) # change the last dimension
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = .5e-3)

## training model function
def trained_model(model, criterion, optimizer, epochs):
    loss_history, acc_history = [],[]
    for epoch in range(epochs):
        
        ## report epoch
        print('Epoch:', str(epoch+1) + '/' + str(epochs))
        print('-'*10)

        ## confusion matrix
        conf_matrix = torch.zeros(4, 4)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train() #this trains the model
            else:
                model.eval() #this evaluates the model

            running_loss, running_corrects = 0.0, 0 
            
            ## adding Y_test and Y_pred
            Y_test=[]
            Y_pred=[]
            Y_pred_prob = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) #convert inputs to cpu or cuda
                labels = labels.to(device) #convert labels to cpu or cuda

                outputs = model(inputs) #outputs is inputs being fed to the model
                loss = criterion(outputs, labels) #outputs are fed into the model
                
                preds_soft = torch.softmax(outputs, dim=1)
                _, preds = torch.max(preds_soft, 1)
                running_loss += loss.item() * inputs.size(0) #loss multiplied by the first dimension of inputs
                running_corrects += torch.sum(preds == labels.data) #sum of all the correct predictions
                
                if phase == 'train':
                    optimizer.zero_grad() #sets gradients to zero
                    loss.backward() #computes sum of gradients
                    optimizer.step() #preforms an optimization step

                ## calculate sensitivity and specificity
                if phase == 'validation':
                    ## update confusion matrix
                    for t, p in zip(labels.data, preds):
                        conf_matrix[t, p] += 1
                
                ## if it is the last epoch record the results
                if phase=='validation' and epoch == epochs-1:
                    y_pred = preds.detach().cpu().numpy()
                    y_test = labels.data.detach().cpu().numpy()
                    y_pred_prob = preds_soft.detach().cpu().numpy()
                    
                    ## append to list
                    Y_pred_prob.append(y_pred_prob)
                    Y_test.append(y_test)
                    Y_pred.append(y_pred)
            
            ## calculate loss
            epoch_loss = running_loss / len(data_images[phase]) #this is the epoch loss
            epoch_accuracy = running_corrects.double() / len(data_images[phase]) #this is the epoch accuracy

            print(phase, ' loss:', epoch_loss, 'epoch_accuracy:', epoch_accuracy)
            
            loss_history.append(epoch_loss)
            acc_history.append(epoch_accuracy)
            
            if phase == 'validation':
                ## need to adjust the index based on the class index
                correct_covid_label = conf_matrix[1,1]
                total_covid_label = torch.sum(conf_matrix[1,:])
                correct_non_covid = conf_matrix[0,0] + conf_matrix[2,2] + conf_matrix[3,3]
                total_non_convid_label = torch.sum(conf_matrix[0,:]) + torch.sum(conf_matrix[2,:]) + torch.sum(conf_matrix[3,:])

                ## report sensitivity and specificity
                print('sensitivity: ', correct_covid_label/total_covid_label)
                print('specificity: ', correct_non_covid/total_non_convid_label)
    
    ## combine result
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    Y_pred_prob =np.concatenate(Y_pred_prob, axis=0)
    
    return model, conf_matrix, Y_test, Y_pred, Y_pred_prob, loss_history, acc_history

## train model
model, mtrx, Y_test, Y_pred, Y_pred_prob, loss_his, acc_his = trained_model(model, criterion, optimizer, 20)

## eval function
def eval_model(model, dataloader):

    model.eval()
    Y_pred = []
    Y_test = []
    Y_pred_prob = []
    for data, target in dataloader:

        data = data.to(device)
        output = model(data)
        preds_soft = torch.softmax(output, dim=1)
        _, preds = torch.max(preds_soft, 1)
        y_pred = preds.detach().cpu().numpy()
        Y_pred.append(y_pred)
        
        y_test = target.data.detach().cpu().numpy()
        Y_test.append(y_test)
        
        y_pred_prob = preds_soft.detach().cpu().numpy()
        Y_pred_prob.append(y_pred_prob)
        #print(target)
        #print(Y_pred[0].shape)
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    Y_pred_prob = np.concatenate(Y_pred_prob, axis=0)
    return Y_pred, Y_test, Y_pred_prob


## test on test set
Y_pred, Y_test, Y_pred_prob = eval_model(model, testloaders['test'])

## final conf matrix
conf_mtrx = np.zeros((4,4), dtype='int')
for t, p in zip(Y_test, Y_pred):
    conf_mtrx[t, p]+=1

## sensitivity & specificity
Y_test_COVID = [1 if i ==1 else 0 for i in Y_test]
Y_pred_COVID = [1 if i == 1 else 0 for i in Y_pred]

total_covid_label = np.sum(Y_test_COVID)
total_non_covid = len(Y_test_COVID) - total_covid_label
correct_covid_label = 0
correct_non_covid = 0
for i, v in enumerate(Y_test_COVID):
    if v==1 and Y_pred_COVID[i] == v:
        correct_covid_label +=1
    elif v==0 and Y_pred_COVID[i] ==v:
        correct_non_covid +=1
    else:
        pass

sensitivity = correct_covid_label/total_covid_label
specificity = correct_non_covid/total_non_covid

## pr curve
prec, recall, c = precision_recall_curve(Y_test_COVID , Y_pred_prob[:,1])

## save res
pd.DataFrame({'precision': prec, 'recall': recall}).to_csv('pc_curve_0420_crop_3.csv', index=False)

## remove necessary dataset
files = glob.glob('/kaggle/working/train/*/*.png')
for f in files:
    os.remove(f)
files = glob.glob('/kaggle/working/test/*/*.png')
for f in files:
    os.remove(f)
files = glob.glob('/kaggle/working/valid/*/*.png')
for f in files:
    os.remove(f)



