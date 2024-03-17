import pandas as pd 
import numpy as np 
import os 
import torch
import sklearn
from tqdm.auto import tqdm
import datetime
import argparse
import random
import collections
import json
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_2d3d import resnet50_2d3d_full

def select_resnet(network):
	param = {'feature_size': 1024}
	if network == 'resnet50':
		model = resnet50_2d3d_full()
	else:
		raise IoError('model type for resnet is wrong.')
	return model, param

class Dissonance(nn.Module):
	# input size = torch.Size([batch, 2, 3, 224, 224]), torch.Size([batch, 2, 13, 87])
	def __init__(self, visual_shape, audio_shape, num_classes, backbone='resnet50', dropout=0.5, num_layers_in_fc_layers=1024):
		super().__init__()
		self.visual_shape = eval(visual_shape)
		self.audio_shape = eval(audio_shape)
		self.num_classes = num_classes
		self.dropout = dropout
		#self.__nFeatures__ = 24
		#self.__nChs__ = 32
		#self.__midChs__ = 32
		self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

			nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

			nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

			nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
			nn.BatchNorm2d(512),
			nn.ReLU(),
		)
		self.netfcaud = nn.Sequential(
			#nn.Linear(512*21, 4096),
			nn.Linear(512*18, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
        )
		self.netcnnlip, self.param = select_resnet(backbone)
		#self.last_duration = int(math.ceil(30 / 4))
		self.last_duration = 1 # implementation fix

		img_dim = self.visual_shape[-1]
		self.last_size = int(math.ceil(img_dim / 32))

		self.netfclip = nn.Sequential(
			nn.Linear(self.param['feature_size']*self.last_size*self.last_size, 4096),
			nn.BatchNorm1d(4096),
			nn.ReLU(),
			nn.Linear(4096, num_layers_in_fc_layers),
		)

		self.final_bn_lip = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_lip.weight.data.fill_(1)
		self.final_bn_lip.bias.data.zero_()

		self.final_fc_lip = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, num_classes))
		self._initialize_weights(self.final_fc_lip)

		self.final_bn_aud = nn.BatchNorm1d(num_layers_in_fc_layers)
		self.final_bn_aud.weight.data.fill_(1)
		self.final_bn_aud.bias.data.zero_()

		self.final_fc_aud = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_layers_in_fc_layers, num_classes))
		self._initialize_weights(self.final_fc_aud)
		self._initialize_weights(self.netcnnaud)
		self._initialize_weights(self.netfcaud)
		self._initialize_weights(self.netfclip)
		# For implementation unification
		self.softmax = nn.Softmax(dim=-1)
	def forward_aud(self, x):
		# input size = torch.Size([batch, 2, 13, 87])
		(B, NF, C, H, W) = x.size() # _ = 1
		x = x.view(B*NF, C, H, W)
		mid = self.netcnnaud(x)
		#BNF, C_o, H_o, W_o = mid.size()
		#mid = mid.view(B, -1, C_o, H_o, W_o).permute(0,2,1,3,4).contiguous()
		#mid = F.avg_pool3d(mid, (NF, 1, 1), stride=(1, 1, 1))
		mid = mid.view((mid.size()[0], -1))
		out = self.netfcaud(mid)
		return out
	def forward_lip(self, x):
		# x: torch.Size([batch, 2, 3, 224, 224])
		#(B, N, C, NF, H, W) = x.shape
		#x = x.view(B*N, C, NF, H, W)
		(B, NF, C, H, W) = x.size()
		#x = x.permute(0,2,1,3,4).contiguous()
		x = x.view(-1, C, H, W).unsqueeze(2)
		feature = self.netcnnlip(x)
		feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))
		feature = feature.view(B*NF, self.param['feature_size'], self.last_size, self.last_size)
		feature = feature.view((feature.size()[0], -1))
		out = self.netfclip(feature)
		return out
	def final_classification_lip(self,feature):
		feature = self.final_bn_lip(feature)
		output = self.final_fc_lip(feature)
		return output
	def final_classification_aud(self,feature):
		feature = self.final_bn_aud(feature)
		output = self.final_fc_aud(feature)
		return output
	def forward_lipfeat(self, x):
		mid = self.netcnnlip(x)
		out = mid.view((mid.size()[0], -1))
		return out
	def _initialize_weights(self, module):
		for m in module:
			if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.ReLU) or isinstance(m,nn.MaxPool2d) or isinstance(m,nn.Dropout):
				pass
			else:
				m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None: m.bias.data.zero_()
	# input size = torch.Size([batch, 2, 3, 224, 224]), torch.Size([batch, 2, 13, 87])
	def forward(self, v_input, a_input, return_feats=False):
		# v_input: torch.Size([batch, 2, 3, 224, 224])
		# a_input: torch.Size([batch, 2, 3, 224, 224])
		# visual encoder: pred_v
		#print(f"v_input: {v_input.size()}\na_input: {a_input.size()}")
		pred_v = self.forward_lip(v_input)
		#print(f"LIP predicted: {pred_v.size()}")
		# audio encoder: pred_a
		pred_a = self.forward_aud(a_input)
		#print(f"AUD predicted: {pred_a.size()}")
		class_v = self.final_classification_lip(pred_v)
		#print(f"LIP class predicted: {class_v.size()}")
		class_a = self.final_classification_aud(pred_a)
		#print(f"AUD class predicted: {class_a.size()}")
		v_softmax = self.softmax(class_v)
		#print(f"LIP softmax: {v_softmax.size()}")
		a_softmax = self.softmax(class_a)
		#print(f"AUD softmax: {a_softmax.size()}")
		if return_feats:
			return v_softmax, a_softmax, pred_v, pred_a
		return v_softmax, a_softmax