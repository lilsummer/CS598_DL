# CS598
Use pretrained models on chest x-ray dataset in order to classify:

* Normal images
* Bacterial pneumonia
* COVID-19
* Non COVID-19 viral pneumonia

## Changelog
Please see here for the changelog

## Notebook
Notebook directory includes all the kaggle notebook used for the model pipeline. This includes

* `resnet-classifier` use pretrained resnet50
* `squeezenet-classifier` 
* `densenet-classifier` use densenet 121
* `1cycle-resnet50` use 1cycle learning rate optimization to fine-tune resnet50


## ReadMe archive
Here is the history of model training

## TODO
* No data leakage from the bacterial dataset (need) (done)
* Increase test size (need) (done)
* without rotation (need)(done)
* Use torch transform plain version 
* color-wise augmentation (need)
* Increase number of images using Augmentor (done)
* loss curve plot (done)
* DenseNet (need) (done)
* learning rate scheduler and 1 cycle learning (done)
* check on other specific aumentation techniques
* Check on bugs in the training function (need) (done)
* stepLR (need) (done)
* Other models (need) (other resNet)
* update the tables (need)
* reduceLRonPlatau section - figure (need)
* reduceLR section - table
* reduceLR result - text
* visualization before & after (need)
* go over project requirement

