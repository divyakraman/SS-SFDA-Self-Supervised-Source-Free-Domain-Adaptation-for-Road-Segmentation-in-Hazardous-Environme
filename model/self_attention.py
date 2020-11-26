import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

affine_par = True


#First layer of DeepLab
class Encoder(nn.Module):
	def __init__(self, deeplab_net):
		super(Encoder, self).__init__()
		self.conv1 = deeplab_net.conv1
		self.bn1 = deeplab_net.bn1
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = deeplab_net.maxpool
		self.layer1 = deeplab_net.layer1
		self.layer2 = deeplab_net.layer2
		self.layer3 = deeplab_net.layer3
	
	def forward(self, image):
		out = self.conv1(image) #/2
		out = self.bn1(out)
		out = self.relu(out)
		out = self.maxpool(out) #/2
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)

		return out
		#256 channel output


class Self_Attn(nn.Module):
	""" Self attention Layer"""
	# From SAGAN paper
	def __init__(self,in_dim):
		super(Self_Attn,self).__init__()
		self.chanel_in = in_dim
		
		self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #
		self.avgpool = nn.AvgPool2d(kernel_size=24, stride=16, padding=1, ceil_mode=True)
	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize,C,width ,height = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,width,height)
		
		out = self.gamma*out + x
		batch_size, attn_size, attn_size = attention.size()
		attention = attention.view(1,1, attn_size, attn_size)
		attention = self.avgpool(attention)
		
		return out,attention


class Classifier_Module(nn.Module):
	def __init__(self, inplanes, dilation_series, padding_series, num_classes):
		super(Classifier_Module, self).__init__()
		self.conv2d_list = nn.ModuleList()
		for dilation, padding in zip(dilation_series, padding_series):
			self.conv2d_list.append(
				nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

		for m in self.conv2d_list:
			m.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.conv2d_list[0](x)
		for i in range(len(self.conv2d_list) - 1):
			out += self.conv2d_list[i + 1](x)
			return out


class Decoder(nn.Module):

	def __init__(self, in_dim, num_classes=1):
		super(Decoder,self).__init__()
		self.chanel_in = in_dim
		
		self.layer5 = self._make_pred_layer(Classifier_Module, in_dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
		
		
	def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
		return block(inplanes, dilation_series, padding_series, num_classes)

	def forward(self,x):
		out = self.layer5(x)
		return out

class MainModule(nn.Module):
	def __init__(self, deeplab_net, num_classes=1):
		super(MainModule,self).__init__()
		self.num_classes = num_classes
		self.encoder = Encoder(deeplab_net)
		self.SA = Self_Attn(in_dim=1024)
		self.decoder = Decoder(in_dim=1024, num_classes = num_classes)
		self.softmax = nn.Softmax()
    

		
	def forward(self,image):
		out = self.encoder(image)
		out_sa, out_sa_attn = self.SA(out)
		del out_sa_attn
		out = self.decoder(out_sa)
		out = self.softmax(out)
		#out_attn = torch.cat((out_sa_attn, out_ga_attn), 1)
		return out, out_sa



	
