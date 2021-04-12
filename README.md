### Paper - [**SS-SFDA: Semi Supervised Source Free Domain Adaptation for Road Segmentation for Road Segmentation in Hazardous Environments**](https://arxiv.org/abs/2012.08939)

Project Page - https://gamma.umd.edu/researchdirections/autonomousdriving/weathersafe/ 

<!--
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/rSPIah0liTA"></iframe>
  </figure>
-->

Watch the video at:
https://user-images.githubusercontent.com/18447610/114395597-33c98800-9bba-11eb-86da-d34e64b4a62f.mp4





<!--[![Watch the video](https://obj.umiacs.umd.edu/gamma-umd-website-imgs/researchdirections/autonomousdriving/SAfE.png)](https://www.youtube.com/watch?v=rSPIah0liTA)-->

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
|Dataset|Dataloader|List of images|
|-----|-----|-----|
|Clear weather CityScapes|dataset/cityscapes.py|dataset/cityscapes_list (train_images, val_images, train_labels, val_images)|
|Synthetic Fog| dataset/cityscapes_fog.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Synthetic Rain | dataset/cityscapes_rain.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Synthetic Rain | dataset/cityscapes_rain.py | dataset/cityscapes_list (train_rain_fog, val_rain_fog) |
|Real Fog - Foggy Zurich| dataset/foggy_zurich/train(test).py | dataset/foggy_zurich/lists_file_names  |
|Real, Night Driving - Dark Zurich | dataset/dark_zurich/train(test).py | dataset/dark_zurich/lists_file_names  |
|Heterogeneous Real, Rain + Night - Raincouver | dataset/raincouver/raincouver.py | dataset/raincouver (train_rain_fog, val_rain_fog) |
|Heterogeneous Real, Berkeley Deep Drive | ADD | ADD |

#### Models
model/drnd38.py - DRN-D-38 model <br>
model/drnd38_attention.py - DRN-D-38 model with self-attention

### Testing a pretrained model
Coming soon ! 


### Visualization
Coming soon!

### Training your own model
Coming soon!

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


## Acknowledgements

This code is heavily borrowed from [**AdaptSegNet**](https://github.com/wasidennis/AdaptSegNet), and [**SAGAN**](https://github.com/heykeetae/Self-Attention-GAN)

