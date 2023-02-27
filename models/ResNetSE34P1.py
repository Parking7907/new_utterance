#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *

class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, context=True,out_bn=True ,**kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.log_input  = log_input
        self.context    = context
        self.out_bn     = out_bn

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        #self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        
        if self.context:
            attn_input = 64 * 3
        else:
            attn_input = 64
        
        if self.encoder_type == "SAP":
            attn_output = 64
        elif self.encoder_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")
        print("self.encoder_type", self.encoder_type, attn_input, attn_output, nOut)
        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

        self.mp3 = nn.MaxPool1d(3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        #print("0", x.shape)# 400, 32240 => B X 33240
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                #print("1", x.shape) # B, 1, 40, 302 => B X 40 X 202
                if self.log_input: x = torch.log(x + 1e-6)
                x = self.instancenorm(x).unsqueeze(1).detach()
                #print("2", x.shape) # B, 1, 40, 302 => B X 1 X 40 X 202

        x = self.conv1(x)
        #print("3", x.shape)# B, 16, 20, 302 => B X 16 X 40 X 202 => B X 32 X 20 X 202
        x = self.bn1(x)
        #print("4", x.shape)# 400, 16, 20, 202 => B X 16 X 20 X 202 => B X 32 X 20 X 202
        x = self.relu(x)
        #print("5", x.shape)# 400, 16, 20, 202 => B X 16 X 20 X 202 => B X 32 X 20 X 202
        x = self.layer1(x)
        #print("6", x.shape)# 400, 16, 20, 202 => B X 16 X 20 X 202 => B X 32 X 20 X 202
        x = self.layer2(x)
        #print("7", x.shape)#400, 32, 10, 101 => B X 32 X 10 X 202 => B X 64 X 10 X 101
        x = self.layer3(x)
        #print("8", x.shape)# 400, 64, 5, 51 => B X 64 X 5 X 51 => B X 128 X 5 X 202
        #x = self.layer4(x)
        #print("9", x.shape)# 400, 128, 5, 51 => B X 64 X 5 X 51 변화점
        x = torch.mean(x, dim=2, keepdim=True)
        #print("10", x.shape)# 400, 64, 1, 51 => B X 128 X 1 X 51

        # Attention 이후 나와야할 결과물 : 400 51 128 X 128 1 => 400, 51 => 400, 128
        x = x.permute(0,1,3,2).squeeze(-1) #400, 384,
        ####################################################### Attention From RawNet ###################################################################
        t = x.size()[-1]
        #print("T", t, x.shape,  torch.mean(x, dim=2).repeat(1, 1, t).shape, torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t).shape)
        #400, 128, 51 / 400, 128, / 400, 128 => T, 51,  B X 64 X 51, 1, 800, 3264, B X 64, 51
        if self.context:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, 1, t),
                ),
                dim=1,
            )
        else:
            global_x = x
        if True in torch.isnan(global_x):
            print("NAN!!")
        #print("Global_", global_x.shape) #400, 384, 51 => B, 192, 51 => B, 384, 51
        w = self.attention(global_x) 
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )
        
        x = torch.cat((mu, sg), 1)
        #print("22", w.shape, mu.shape, sg.shape,x.shape, w.shape) # B X 128 X 51,  B X 128, B X 128, B X 128 X 51, B X 1536 X 319 [400Frame: 64 X 1536, 64 X 1536, 64 X 1536 X 426, 64 X 1536 X 426]
        #B 128 51, B 128, B 128, B 256, B 128 51
        #print("4", x.shape) #B,256 [400Frame: B X 3072] B 256
        x = self.bn5(x)
        #print("5", x.shape) #B X 256 [400Frame: B X 3072] B 256
        x = self.fc6(x)
        #print("6", x.shape) #B X 512 [400Frame: B X 256] B 256
        if self.out_bn:
            x = self.bn6(x)
        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64]
    model = ResNetSE(SEBasicBlock, [3, 4, 6], num_filters, nOut, **kwargs)
    return model
