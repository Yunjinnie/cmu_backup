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
import torch.nn.functional as F



class deepfake_v_model(torch.nn.Module):
    # input size = torch.Size([batch, 2, 25, 3, 224, 224])
    # input 2개의 이미지를 받아서 crossattention을 적용한 후 fc layer를 거쳐 class를 분류하는 모델
    def __init__(self, input_shape, num_classes, backbone="resnet50", dropout=0.5):
        # print("input_shape: ", input_shape)

        super(deepfake_v_model, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.dropout = dropout
        self.conv1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # test backbone: resnet50
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x: torch.Size([batch, 2, 3, 224, 224])
        batch_size = x.shape[0]
        num_clip = x.shape[1]
        num_channel = x.shape[2]
        height = x.shape[3]
        width = x.shape[4]
        
        # pre_x: query, next_x: key, value
        pre_x = x[:, 0, :, :, :]
        next_x = x[:, 1, :, :, :]

        # x: torch.Size([batch,3, 224, 224]) -> torch.Size([batch,512])
        pre_x = self.backbone(pre_x)
        next_x = self.backbone(next_x)

        # 1d conv

        # x: torch.Size([batch,512]) -> torch.Size([batch, 1, 512])
        pre_x = pre_x.unsqueeze(1)
        next_x = next_x.unsqueeze(1)

        # pre_x, _ = self.cross_attention(pre_x, next_x, next_x)
        next_x, _ = self.cross_attention(next_x, pre_x, pre_x)

        # x: torch.Size([batch, 1, 512]) -> torch.Size([batch, 512])
        x_rep = next_x.squeeze(1)

        # x: torch.Size([batch, 512]) -> torch.Size([batch, num_classes])
        x = self.fc(x_rep)
        x = self.softmax(x)

        return x, x_rep



class deepfake_a_model(torch.nn.Module):
    # input size = torch.Size([batch, 2, 13, 87])
    # input 2개의 mfcc를 받아서 crossattention을 적용한 후 fc layer를 거쳐 class를 분류하는 모델
    def __init__(self, input_shape, num_classes, backbone="resnet50", dropout=0.5):
        # print("input_shape: ", input_shape)

        super(deepfake_a_model, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.dropout = dropout
        # stride 1d conv, padding same
        self.conv1 = torch.nn.Conv1d(13, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # mfcc test backbone: resnet50
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(512, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x: torch.Size([batch, 2, 3, 224,224])
        
        # pre_x: query, next_x: key, value
        pre_x = x[:, 0, :, :, :]
        next_x = x[:, 1, :, :, :]

        # x: torch.Size([batch,13, 87]) -> torch.Size([batch,64, 87])
        # pre_x = self.conv1(pre_x)
        # next_x = self.conv1(next_x)

        # x: torch.Size([batch, 87, 64]) -> torch.Size([batch, 87, 512])
        pre_x = self.backbone(pre_x)
        next_x = self.backbone(next_x)

        # x: torch.Size([batch,512]) -> torch.Size([batch, 1, 512])
        pre_x = pre_x.unsqueeze(1)
        next_x = next_x.unsqueeze(1)

        # pre_x, _ = self.cross_attention(pre_x, next_x, next_x)
        next_x, _ = self.cross_attention(next_x, pre_x, pre_x)

        # x: torch.Size([batch, 1, 512]) -> torch.Size([batch, 512])

        x_rep = next_x.squeeze(1)
        
        # x: torch.Size([batch, 512]) -> torch.Size([batch, num_classes])
        x = self.fc(x_rep)
        x = self.softmax(x)

        return x, x_rep


# visual + audio model + av multimodal model 
# visual output : class
# audio output : class
# av 는 visual과 audio의 output을 concat해서 fc layer를 거쳐 class를 분류하는 모델
# av는 score만 visual, audio에게 전달하고, visual, audio는 각각 class를 분류하는 모델
class SmdaNet(torch.nn.Module):
    # input size = torch.Size([batch, 2, 3, 224, 224]), torch.Size([batch, 2, 13, 87])
    # input 2개의 이미지와 mfcc를 받아서 crossattention을 적용한 후 fc layer를 거쳐 class를 분류하는 모델
    def __init__(self, visual_shape, audio_shape, num_classes, backbone="resnet50", dropout=0.5):

        super(SmdaNet, self).__init__()
        self.visual_shape = visual_shape
        self.audio_shape = audio_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.dropout = dropout
        self.unimodal_weight = 0.5


        # test backbone: resnet50
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = torch.nn.Linear(2048, 512)

        self.deepfake_v_model = deepfake_v_model(visual_shape, num_classes, backbone, dropout)
        self.deepfake_a_model = deepfake_a_model(audio_shape, num_classes, backbone, dropout)

        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.fc = torch.nn.Linear(1536, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, v_input, a_input):
        # v_input: torch.Size([batch, 2, 3, 224, 224])
        # a_input: torch.Size([batch, 2, 3, 224, 224])

        # visual encoder: pred_v - softmax(4), v_rep - representation(512)
        pred_v, v_rep = self.deepfake_v_model(v_input)
        # audio encoder: pred_a - softmax(4), a_rep - representation(512)
        pred_a, a_rep = self.deepfake_a_model(a_input)

        v_rep = v_rep.unsqueeze(1)
        a_rep = a_rep.unsqueeze(1)
        
        # v-a cross
        va_rep, _ = self.cross_attention(v_rep, a_rep, a_rep)
        # a-v cross
        av_rep, _ = self.cross_attention(a_rep, v_rep, v_rep)

        # a+v+va concat
        va_concat = torch.cat((v_rep, a_rep, va_rep), dim=2)
        av_concat = torch.cat((v_rep, a_rep, av_rep), dim=2)

        # visual, va_detector: weighted average
        va_concat = va_concat.squeeze(1)
        av_concat = av_concat.squeeze(1)

        # audio, av_detector: weighted average
        va_softmax = self.softmax(self.fc(va_concat))
        weigted_v_softmax = self.unimodal_weight * pred_v + (1 - self.unimodal_weight) * va_softmax        

        av_softmax = self.softmax(self.fc(av_concat))
        weigted_a_softmax = self.unimodal_weight * pred_a + (1 - self.unimodal_weight) * av_softmax

        return weigted_v_softmax, weigted_a_softmax