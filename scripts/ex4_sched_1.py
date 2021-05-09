## from processed data to model results
## this excldues preprossessing_data_set2.py
## !pip install catalyst


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
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import utils


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

## use catalyst

runner = SupervisedRunner()


## model code
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss =F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        #acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        #batch_accs = [x['val_acc'] for x in outputs]
        #epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch,  result['train_loss'], result['val_loss']))

class XrayRES(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50()
        ## last fully-connected
        num_ftrs = self.network.fc.in_features
        self.network.fc =  nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4))
    def forward(self, xb):
        return torch.softmax(self.network(xb), dim=1)
    
    def freeze(self):
        # to freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        
        # only train the last layer
#         for param in self.network.fc.parameters():
#             param.require_grad = True
            
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
model = to_device(XrayRES(4), device)

## use runner
optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

os.mkdir('/kaggle/working/log')
runner.train(
    model=model,
    logdir='/kaggle/working/log/',
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=dataloaders,
    num_epochs=20,
    verbose=False
)

## plot metrics
logdir='/kaggle/working/log/'
utils.plot_metrics(logdir, metrics=['loss'])
utils.plot_metrics(logdir, metrics=['lr'])


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
