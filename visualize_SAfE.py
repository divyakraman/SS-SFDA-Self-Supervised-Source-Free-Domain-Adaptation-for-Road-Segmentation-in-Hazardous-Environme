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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dtype = torch.cuda.FloatTensor #GPU

data_dir = 'path' 
data_list = './dataset/cityscapes_list/val_rain_fog.txt'
batch_size = 1
num_steps = 40
input_size_target = '1024, 512'
eval_set = 'train'
num_workers = 1

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

w, h = map(int, input_size_target.split(','))
input_size_target = (w, h)

targetloader = data.DataLoader(cityscapesRainDataSet(data_dir, data_list, max_iters=num_steps * batch_size, crop_size=input_size_target, 
    scale=False, mean=IMG_MEAN, set = 'train'), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

targetloader_iter = enumerate(targetloader)
num_classes = 2

colors = [ [128,64,128],
[0,0,0]
]


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


hist = np.zeros((num_classes,num_classes))

for iteration in range(0,num_steps):
    _, batch = targetloader_iter.__next__()
    images, labels = batch
    images = Variable(images).cuda()


    pred, out_attn = model(images)
    out_attn = interp_target(out_attn[:,:,:,:])
    out_attn = out_attn[0,0,:,:]
    out_attn = out_attn.detach()
    out_attn = out_attn.cpu()
    out_attn = out_attn.numpy()
    
    pred = interp_target(pred)
    pred_entropy = prob_2_entropy(pred)
    pred_entropy = pred_entropy.detach()
    pred_entropy = pred_entropy.cpu()
    pred_entropy = pred_entropy.numpy()
    pred_entropy = pred_entropy[0,:,:,:]
    pred_entropy = np.sum(pred_entropy, axis=0)
    
            
    pred = pred.detach()
    pred = pred.cpu()
    pred = pred.numpy()
    pred = pred[0,:,:,:]
    pred_road = pred[0,:,:]
    pred = np.argmax(pred,0)
    labels = labels.cpu()
    labels = labels.numpy()
    labels = labels[0,:,:]
    labels[labels!=0]=1
    labels = labels.astype(np.int)
    hist += fast_hist(labels.flatten(), pred.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

    torch.cuda.empty_cache() #clear cached memory
    print(iteration)

    
    pred_color_labels = np.zeros((512,1024,3))
    for i in range(len(colors)-1):
        pred_color_labels[np.where(pred==i)]=colors[i]
    
    #Save drivable area results
    image_name = 'Results/Rain100mm/'+str(iteration)+'_pred'+'.jpg'
    pred_color_labels = pred_color_labels/255.0
    plt.imsave(image_name,pred_color_labels)
    #Save image
    image = images[0,0,:,:].cpu()
    image = image.numpy()
    image_name = 'Results/Rain100mm/'+str(iteration)+'_image'+'.jpg'
    plt.imsave(image_name,image)
    #Save GT
    pred_color_labels = np.zeros((512,1024,3))
    for i in range(len(colors)-1):
        pred_color_labels[np.where(labels==i)]=colors[i]
    image_name = 'Results/Rain100mm/'+str(iteration)+'_GT'+'.jpg'
    pred_color_labels = pred_color_labels/255.0
    plt.imsave(image_name,pred_color_labels)
    #Save extent of safety
    image_name = 'Results/Rain100mm/' + str(iteration)+'_road_heatmap'+'.jpg'
    plt.imsave(image_name,pred_road,cmap='viridis')
    #Save entropy
    image_name = 'Results/Rain100mm/'+str(iteration)+'_entropy'+'.jpg'
    plt.imsave(image_name,pred_entropy,cmap='viridis')


    
    
    print("saved result")
    



mIoUs = per_class_iu(hist)

for ind_class in range(num_classes):
    print('===> Class '+str(ind_class)+':\t'+str(round(mIoUs[ind_class] * 100, 2)))


print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

print('===> Accuracy Overall: ' + str(np.diag(hist).sum() / hist.sum() * 100))
#acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) 

#for ind_class in range(num_classes):
#    print('===> Class '+str(ind_class)+':\t'+str(round(acc_percls[ind_class] * 100, 2)))






