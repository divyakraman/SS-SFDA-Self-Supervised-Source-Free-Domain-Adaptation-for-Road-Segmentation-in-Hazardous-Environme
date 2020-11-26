import numpy as np
import matplotlib.pyplot as plt
import glob
#import imageio
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import os.path as osp
from dataset.cityscapes_rain import cityscapesRainDataSet
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dtype = torch.cuda.FloatTensor #GPU


data_dir = 'path' 
data_list = './dataset/cityscapes_list/val_rain_fog.txt'
batch_size = 1
num_steps = 359
#num_steps = 5
input_size_target = '1024, 512'
eval_set = 'train' #For rain and fog, val images taken from original train set only (Original train set split into train and val)
num_workers = 1

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

w, h = map(int, input_size_target.split(','))
input_size_target = (w, h)

targetloader = data.DataLoader(cityscapesRainDataSet(data_dir, data_list, max_iters=num_steps * batch_size, crop_size=input_size_target, 
    scale=False, mean=IMG_MEAN, set = 'train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

targetloader_iter = enumerate(targetloader)
num_classes = 2

colors = [ [128,64,128],
[0,0,0]]
#ignoring void class; road and not road

interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

def fast_hist(a,b,n):
    k = (a>=0) & (a<n)
    return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
    return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


model = torch.load('model.pth')
model = model.cuda()
tp = 0
fp = 0
fn = 0

hist = np.zeros((num_classes,num_classes))

conf_matrix = np.array([[0,0],[0,0]])

for iteration in range(0,num_steps):
    _, batch = targetloader_iter.__next__()
    images, labels = batch
    images = Variable(images).cuda()

    out, out_attn = model(images)

    pred = interp_target(out)
    del out
    


    
            
    pred = pred.detach()
    pred = pred.cpu()
    pred = pred.numpy()
    pred = pred[0,:,:,:]
    pred = np.argmax(pred,0)
    labels = labels.cpu()
    labels = labels.numpy()
    labels1 = labels
    del labels
    labels1[labels1!=0]=1
    labels1 = labels1[0,:,:]
    labels1 = labels1.astype(np.int)
    hist += fast_hist(labels1.flatten(), pred.flatten(), num_classes)

    conf_matrix = conf_matrix + confusion_matrix(np.ravel(labels1),np.ravel(pred))


    mIoUs = per_class_iu(hist)
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

    torch.cuda.empty_cache() #clear cached memory
    print(iteration)

mIoUs = per_class_iu(hist)

conf_matrix = conf_matrix/(512*1024*500)

for ind_class in range(num_classes):
    print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))


print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

print('===> Accuracy Overall: ' + str(np.diag(hist).sum() / hist.sum() * 100))
acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) 

for ind_class in range(num_classes):
    print('===> Class '+str(ind_class)+':\t'+str(round(acc_percls[ind_class] * 100, 2)))

#tn,fp,fn,tp = conf_matrix.ravel() if 1:positive, 0:negative
tp,fn,fp,tn = conf_matrix.ravel() 
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print('===> Precision: ' + str(round(np.nanmean(precision), 2)))
print('===> Recall: ' + str(round(np.nanmean(recall), 2)))
print('===> F1: ' + str(round(np.nanmean(f1), 2)))




