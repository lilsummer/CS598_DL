# CS598

### 2020-04-12
Model ResNet50
Data: same as previous except that we kept only one image from each patient so there's no data leakage
#### Data Augmentation
Same as before
#### Training
same as before; time 6349 seconds
#### Testing metric
pr curve saved in pr_curve_0412_b_removal.csv

confusion matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 253        | 1     | 6      | 28    |
| COVID    | 1          | 664   | 57     | 2     |
| Normal   | 0          | 19    | 2001   | 19    |
| Viral    | 46         | 0     | 9      | 214   |

sensitivity specificity
```
0.9171, 0.9507
```
### 2020-04-13
Model SqueezeNet1_0
Data: same as previous
#### Data Augmentation
Same as before
#### Training
Same as before; time 5115 seconds
#### Testing metric
pr curve saved in pr_curve_0413_squeezenet10.csv

Confusion matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 250        | 0     | 6      | 32    |
| COVID    | 3          | 616   | 103    | 2     |
| Normal   | 3          | 39    | 1989   | 8     |
| Viral    | 71         | 0     | 9      | 189   |

sensitivity specificity
```
0.8508, 0.9353
```
### 2020-04-13
Model squeezeNet1_0
#### Data Augmentation
removed the rotation on training data
#### Training
training time 3708 seconds
#### Testing metric
pr curve saved in pr_curve_0413_squeezenet10_b.csv
Confusion matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 193        | 1     | 8      | 86    |
| COVID    | 0          | 698   | 25     | 1     |
| Normal   | 0          | 165   | 1863   | 11    |
| Viral    | 25         | 6     | 5      | 233   |

sensitivity specificity
```
0.9641, 0.8817
```
### 2020-04-13
Model ResNet50
#### Data Augmentation
remove rotation
#### Training
Training time 4849 seconds
#### Testing metric
pr curve saved in pr_curve_0413_resnet_b.csv
notebook version: classifier on COVID version 12

sensitivity specificity
```
0.9627	0.8925
```

### 2020-04-14
Model ResNet50
#### Data Augmentation
Add random distortion
#### Training 
Training time 4563 seconds
#### Testing metric
pr curve saved in pr_curve_0414_resnet_distortion.csv
notebook version: classifier on COVID version 13

sensitivity sepcificity
```
0.924	0.9095
```

### 2020-04-15
Model ResNet50
#### Data Augmentation
same as before, using Augmentor to increase the size of the training COVID data to 4000 images
#### Training
lr = .5e-3 time:3162 seconds
#### Testing metric
pr curve saved in pc_curve_0414_resnet_augmentor4000_lr.csv
notebook version: classifier on COVID version 15

sensitivity specificity
```
0.9199, 0.8987
``


### TODO
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
