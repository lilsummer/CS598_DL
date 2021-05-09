## Exploration on Data Augmentation and Learning Rate Optimization on Detection of COVID-19 via Chest X-ray (CXR) Images

Use pretrained models on chest x-ray dataset in order to classify:

* Normal images
* Bacterial pneumonia
* COVID-19
* Non COVID-19 viral pneumonia

The pretrained models include:

* ResNet50
* SqueezeNet
* DenseNet121

## Table of contents
* [Usage](#usage)
* [Scripts](#scripts)
* [Changelog](#changelog)
* [Notebook](#notebook)
* [ReadMeArchieve](#readmearchieve)

## Usage
This project was implemented on Kaggle Notebook (GPU kernel). The best way to run the experiements is to upload the notebook in `/notebook/` directory to [kaggle notebook](), and link two following dataset:

* [Kaggle COVID-19 radiography dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
* [Kaggle Chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The notebook will perform the following steps end-to-end

* data preprocessing
* model training and inferencing
* metrics reporting

## Scripts
`/scripts/` folder includes:

* preprocess_data_set2.py
* ex2_pipe_1.py
* ex3_sched_1.py
* ex4_sched_1.py
* ex5.py

## Changelog
Please see [here](https://github.com/lilsummer/CS598_DL/blob/main/changelog.md) for the changelog

## Notebook
Notebook directory includes all the kaggle notebook used for the model pipeline. This includes

* `resnet-classifier` use pretrained resnet50
* `squeezenet-classifier` 
* `densenet-classifier` use densenet 121
* `1cycle-resnet50` use 1cycle learning rate optimization to fine-tune resnet50

## ReadMeArchieve
Here is the history of model training