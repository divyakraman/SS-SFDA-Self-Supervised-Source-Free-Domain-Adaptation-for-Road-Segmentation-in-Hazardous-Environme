### Paper - [**SS-SFDA: Semi Supervised Source Free Domain Adaptation for Road Segmentation for Road Segmentation in Hazardous Environments**](https://arxiv.org/abs/2012.08939)

Project Page - https://gamma.umd.edu/researchdirections/autonomousdriving/weathersafe/

Please cite our paper if you find it useful.

```
@article{kothandaraman2020safe,
  title={SAfE: Self-Attention Based Unsupervised Road Safety Classification in Hazardous Environments},
  author={Kothandaraman, Divya and Chandra, Rohan and Manocha, Dinesh},
  journal={arXiv preprint arXiv:2012.08939},
  year={2020}
}
```

<p align="center">
<img src="cover_pic.png" width="360">
</p>

Table of Contents
=================
 * [Paper - <a href="link to paper" rel="nofollow"><strong>SS-SFDA: Semi Supervised Source Free Domain Adaptation for Road Segmentation for Road Segmentation in Hazardous Environments</strong></a>](#paper---SS-SFDA-Semi-Supervised-Source-Free-Domain-Adaptation-for-Road-Segmentation-for-Road-Segmentation-in-Hazardous-Environments)
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

#### Models

#### Utils

### Testing a pretrained model


### Visualization


### Training your own model


### Datasets
* [**Clear weather: CityScapes**](https://www.cityscapes-dataset.com/) 
* [**Synthetic: Rain and Fog**](https://team.inria.fr/rits/computer-vision/weather-augment/)  
* [**Real dataset (night): Dark Zurich**](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)
* [**Real dataset (fog): Foggy Zurich**](http://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/)
* [**Heterogeneous Real dataset (rain+night): Raincouver**](https://www.cs.ubc.ca/~ftung/raincouver/index.html) 
* [**Heterogeneous Real dataset (multiple weather, lighting conditions): Berkeley Deep Drive**](https://bdd-data.berkeley.edu/) 


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

