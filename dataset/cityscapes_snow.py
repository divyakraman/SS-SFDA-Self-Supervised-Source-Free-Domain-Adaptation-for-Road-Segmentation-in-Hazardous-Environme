import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2

class cityscapesSnowDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.id_to_trainid = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255,
                              6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3,
                              13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 
                              29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255}
        
        self.trainid_to_snowid = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,
                              6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13,
                              13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20} #incorporating snow
        
        #snow point value and random snow on road value
        if(self.list_path[0]=='a' or self.list_path[0]=='b' or self.list_path[0]=='c' or self.list_path[0]=='d'):
            self.snow_point = 140
            self.random_snow_on_road = 0.1
        elif(self.list_path[0]=='e' or self.list_path[0]=='f' or self.list_path[0]=='h' or self.list_path[0]=='j'):
            self.snow_point = 130
            self.random_snow_on_road = 0.085
        elif(self.list_path[0]=='k' or self.list_path[0]=='l' or self.list_path[0]=='m' or self.list_path[0]=='s'):
            self.snow_point = 120
            self.random_snow_on_road = 0.12
        elif(self.list_path[0]=='t' or self.list_path[0]=='u' or self.list_path[0]=='w' or self.list_path[0]=='z'):
            self.snow_point = 150
            self.random_snow_on_road = 0.15




        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_name = name[:-15] + 'gtFine_labelIds.png'
            label_root = self.root
            #label_root = label_root[:-25]
            label_file = osp.join(label_root, "gtFine/%s/%s" % (self.set, label_name))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

          
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image)

        #add_snow(image):    
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
        image_HLS = np.array(image_HLS, dtype = np.float64)     
        brightness_coefficient = 2.5     
        snow_point=140 ## increase this for more snow    
        image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)    
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255    
        image_HLS = np.array(image_HLS, dtype = np.uint8)    
        image_transformed = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
        del image_HLS
        

        image = np.asarray(image_transformed, np.float32)
        del image_transformed
        label = np.asarray(label, np.float32)

        #road_indices = np.argwhere((GT_original == [127, 63, 128]).all(axis=2))
        road_indices = np.argwhere(label == 0)
        rand_indices = np.random.randint(0,len(road_indices), size=int(len(road_indices)*0.1))
        image[road_indices[rand_indices,0], road_indices[rand_indices,1], :] = [255,250,250]
        #GT_original[road_indices[rand_indices,0], road_indices[rand_indices,1], :] = [255,250,250]
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        del label

        label = 255 * np.ones(label_copy.shape, dtype=np.float32)
        for k, v in self.trainid_to_snowid.items():
            label[label_copy == k] = v
        del label_copy

        label[road_indices[rand_indices,0], road_indices[rand_indices,1]] = 1 #snow roads

        
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy()

        
if __name__ == '__main__':
    dst = cityscapesDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()


#train: aachen, bochum, bremen, cologne, darmstadt, dusseldorf, erfurt, hamburg, hanover, jena, krefeld, moncheng, 
#strasbourg, stuttgart, tubingen, ulm, weimar, zurich
#eval: frankfurt, lindau, munstar