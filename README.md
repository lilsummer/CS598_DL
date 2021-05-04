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
* You should also introduce PR-AUC score in section 2.3. (need) (done)
* visualization before & after: use images that have identified region of infection (need)
* the need to align images (add to discussion)
* distortion without cropping (for three models)
* It is unpredictable how the model will perform for larger unseen images. (add this point to discussion)
* Add AUC as metric (need) (done)
* incrementally unfreezing the pretrained models (add this to discussion)
* actual cost of x-ray imgaes (add to intro/discussion)
* other pretrained models on x-ray images (add to discussion)
* explain why validation loss was flatterned (add to result)
* Improve the readme format (need)
* code formulation, packaging (need)
* make sure the repo is upto date (need)
* Hypothesis (need)
* go over project requirement
* data augmentation
* biomarks from x-ray images discussion (done)

