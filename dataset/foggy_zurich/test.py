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

class ZurichFogDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
        self.root = root
        self.label_root = root
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

                
        
        for name in self.img_ids:
            img_file = self.root + '/' + name
            #img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_name = 'gt_labelTrainIds'+ name[3:]
            label_file = self.label_root + '/' + label_name
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

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        
        # re-assign labels to match the format of Cityscapes
        #label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        #for k, v in self.id_to_trainid.items():
        #    label_copy[label == k] = v

        

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), name
