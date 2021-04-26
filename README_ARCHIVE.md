### 2020-04-03
Model: ResNet50
Data: 

|       | COVID | Bateria Pneumonia | Viral Pneumoina | Normal |
|-------|-------|-------------------|-----------------|--------|
| Train | 2892  | 2224              | 1076            | 8153   |
| Test  | 724   | 556               | 269             | 2039   |

#### Data Augmentation
```
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.RandomResizedCrop((244, 244)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(),
        transforms.ToTensor(),
        normalizer
    ]),
    'validation': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop((244, 244)),
        transforms.ToTensor(),
        normalizer
    ])
}
```
#### Training
batch size 32, 20 epochs, 4330 seconds for the entire notebook
learning rate default: 1e-3

#### Testing Metric
PR curve, see figure 101
Confusion matrix

|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 518        | 1     | 19     | 18    |
| COVID    | 3          | 607   | 114    | 0     |
| Normal   | 20         | 62    | 1952   | 5     |
| Viral    | 131        | 4     | 17     | 117   |

Sensitivity & Specificity
```
(0.8384, 0.9033)
```
### 2020-04-05
Model ResNet50
Data: same as previous
**This time adding random sampling for training and testing data**
#### Data Augmentation
Same as before
#### Training
Batch size 32, 25 epochs, lr = 2e-3, 5490 seconds
#### Testing Metric
Confusion Matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 544        | 5     | 5      | 2     |
| COVID    | 1          | 627   | 96     | 0     |
| Normal   | 38         | 55    | 1944   | 2     |
| Viral    | 174        | 3     | 19     | 73    |

sensitivity and specificity

```
0.8660, 0.8942
```

### 2020-04-10
Model ResNet50
Data: same as previous
**This time added methods to save pr-curve data**
#### Data Augmentation
Same as before
#### Training
Batch size 32, 25 epochs, lr = 2e-3, 5490 seconds
#### Testing Metric
PR curve saved in csv file
Confusion Matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 536        | 0     | 17     | 3     |
| COVID    | 11         | 535   | 178    | 0     |
| Normal   | 49         | 18    | 1972   | 0     |
| Viral    | 194        | 0     | 22     | 53    |

sensitivity and specificity

```
0.7390, 0.8942
```

### 2020-04-11
Model ResNet50
Data: same as previous
#### Data Augmentation
Used Augmentor and rotation for training dataset
* Training set pipeline
```
p.resize(width=244, height=244, probability=1)
p.crop_by_size(width=244, height=244, probability=.5)
p.rotate(max_left_rotation=25, max_right_rotation=25, probability=.5)
'train': transforms.Compose([
        p.torch_transform(),
        transforms.ToTensor(),
        normalizer
    ]),
```

* Testing set pipeline
```
p2 = Augmentor.Pipeline()
p2.resize(width=244, height=244, probability=1)
p2.crop_centre(probability=.5, percentage_area=.9)
p2.resize(width=244, height=244, probability=1)
'validation': transforms.Compose([
        p2.torch_transform(),
        transforms.ToTensor(),
        normalizer
    ])
```

#### Training
same as before, training time 6973 seconds

#### Testing metric
PR curve saved in pr_curve_0411.csv
tensor([[5.2000e+02, 2.0000e+00, 1.0000e+00, 3.3000e+01],
        [1.0000e+00, 6.8200e+02, 3.9000e+01, 2.0000e+00],
        [1.0000e+00, 5.0000e+01, 1.9790e+03, 9.0000e+00],
        [5.8000e+01, 1.0000e+00, 1.7000e+01, 1.9300e+02]])

Confusion matrix
|          | prediction |       |        |       |
|----------|------------|-------|--------|-------|
|          | Bacteria   | COVID | Normal | Viral |
| Bacteria | 520        | 2     | 1      | 33    |
| COVID    | 1          | 682   | 39     | 2     |
| Normal   | 1          | 50    | 1979   | 9     |
| Viral    | 58         | 1     | 17     | 193   |


Sensitivity specificity

```
0.9420, 0.9399
```


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