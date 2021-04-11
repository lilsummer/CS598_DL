# CS598
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
Used different normalization


### TODO
* Learning rate (need)(done)
* Train more epoch (done)
* Save PR curve statistics (need)(done)
* Data Augmentation by adding more images
* Random sampling for training and testing (need)(done)
* Augmentor (need) (done)
* Other models (need)
* Other normalization
* No data leakage from the bacterial dataset (need)
