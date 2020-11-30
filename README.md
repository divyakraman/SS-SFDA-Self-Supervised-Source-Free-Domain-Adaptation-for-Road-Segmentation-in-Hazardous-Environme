### Under construction

### Paper - [**SAfE: Self-Attention Based Unsupervised Road Safety Classification in Hazardous Environments**](link to paper)

Project Page - https://gamma.umd.edu/link

Please cite our paper if you find it useful.

```
citation
```

<p align="center">
<img src="cover_pic.png" width="360">
</p>

Table of Contents
=================
 * [Paper - <a href="link to paper" rel="nofollow"><strong>SAfE: Self-Attention Based Unsupervised Road Safety Classification in Hazardous Environments</strong></a>](#paper---safe-self-attention-based-unsupervised-road-safety-classification-in-hazardous-environments)
  * [**Repo Details and Contents**](#repo-details-and-contents)
     * [Code structure](#code-structure)
     * [Testing a pretrained model](#testing-a-pretrained-model)
     * [Training your own model](#training-your-own-model)
     * [Datasets](#datasets)
     * [Dependencies](#dependencies)
  * [**Our network**](#our-network)
  * [**Acknowledgements**](#acknowledgements)

## Repo Details and Contents
Python version: 3.7

### Code structure
#### Dataloaders <br>
The 'dataset' folder contains dataloaders for CityScapes, CityScapes Snow, CityScapes Rain, CityScapes Fog, and the corresponding train-test image splits
#### Models
The 'model' folder contains network architectures for the self-attention based model, and discriminators (with spectral normalization)
#### Utils
Contains the cross entropy loss function

### Testing a pretrained model

Use the code eval_SAfE.py to test a pre-trained model. The path to the model can be set at line 63, and the dataset can be specified at lines 23-30. The code computes the Intersection over Union (IoU), mean IoU, accuracy, mean accuracy, precision, recall and F1 score.

### Visualization

Use the code visualize_SAfE.py to visualize the predictions, attention maps, entropy maps and the heat maps.

### Training your own model

Use the code train_SAfE.py to train your model. The clear weather dataset can be specified at lines 35-37, and the hazardous weather dataset can be specified at lines 38-41. Make sure to load the appropriate dataloaders (lines 226-243, 23-24)

### Datasets
* [**CityScapes**](https://www.cityscapes-dataset.com/) 
* [**CityScapes Rain and Fog**](https://team.inria.fr/rits/computer-vision/weather-augment/) 

### Dependencies
PyTorch <br>
NumPy <br>
SciPy <br>
Matplotlib <br>

## Our network

<p align="center">
<img src="main_architecture.png">
</p>

## Acknowledgements

This code is heavily borrowed from [**AdaptSegNet**](https://github.com/wasidennis/AdaptSegNet), and [**SAGAN**](https://github.com/heykeetae/Self-Attention-GAN)

