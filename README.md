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
* Learning rate (need)(done)
* Train more epoch (done)
* Save PR curve statistics (need)(done)
* Data Augmentation by adding more images (done)
* Random sampling for training and testing (need)(done)
* Augmentor (need) (done)
* Other models (need) (squeezeNet)
* Other normalization (need)
* No data leakage from the bacterial dataset (need) (done)
* Increase test size (need)
* without rotation (need)(done)
* visualization before & after (need)
* Use torch transform plain version 
* color-wise augmentation (need)
* Increase number of images using Augmentor (done)
* loss curve plot (done)
* DenseNet (need) (done)
* smaller learning rate (1e-4) with more epochs (need)
* learning rate scheduler and 1 cycle learning
* check on other specific aumentation techniques
* Check on bugs in the training function (need)
