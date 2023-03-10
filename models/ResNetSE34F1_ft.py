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
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels)

        
        if self.context:
            attn_input = 128 * 3
        else:
            attn_input = 128
        
        if self.encoder_type == "SAP":
            attn_output = 128
        elif self.encoder_type == "ASP":
            attn_output = 1
        else:
            raise ValueError("Undefined encoder")
        print("self.encoder_type", self.encoder_type, attn_input, attn_output, nOut)
        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        '''
        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
            print(self.encoder_type, out_dim, num_filters, block.expansion) #SAP, 128, [16, 32, 64, 128]
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
            print(self.encoder_type, out_dim, num_filters)
        else:
            raise ValueError('Undefined encoder')
        
        self.fc = nn.Linear(out_dim, nOut)
        '''
        
        self.bn5 = nn.BatchNorm1d(256)

        self.fc6 = nn.Linear(256, nOut)
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
        #print("0", x.shape) 400, 32240
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x)+1e-6
                #print("1", x.shape) # B, 1, 40, 302
                if self.log_input: x = torch.log(x + 1e-6)
                x = self.instancenorm(x).unsqueeze(1).detach()
                #print("2", x.shape) # B, 1, 40, 302

        x = self.conv1(x)
        #print("3", x.shape)# B, 16, 20, 302
        x = self.bn1(x)
        #print("4", x.shape) 400, 16, 20, 202
        x = self.relu(x)
        #print("5", x.shape) 400, 16, 20, 202
        x = self.layer1(x)
        #print("6", x.shape) 400, 16, 20, 202
        x = self.layer2(x)
        #print("7", x.shape)400, 32, 10, 101
        x = self.layer3(x)
        #print("8", x.shape) 400, 64, 5, 51
        x = self.layer4(x)
        #print("9", x.shape) 400, 128, 5, 51
        x = torch.mean(x, dim=2, keepdim=True)
        #print("10", x.shape) 400, 128, 1, 51

        # Attention ?????? ???????????? ????????? : 400 51 128 X 128 1 => 400, 51 => 400, 128
        x = x.permute(0,1,3,2).squeeze(-1) #400, 384,
        ####################################################### Attention From RawNet ###################################################################
        t = x.size()[-1]
        f = x.size()[-2]
        #print("T", f, x.shape,  torch.mean(x, dim=1).repeat(1, f, 1).shape, torch.sqrt(torch.var(x, dim=1, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, f, 1).shape)
        #400, 128, 51 / 400, 128, / 400, 128 /
        # F ?????? ?????? : 51 /  B 128 51 / 1, B, 6528 / B, 128, 51 / 
        # F ?????? ?????? : 128 / 400, 128, 51 / 1, 51200, 51 / 400, 128, 51
        if self.context:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=1, keepdim=True).repeat(1, f, 1),
                    torch.sqrt(
                        torch.var(x, dim=1, keepdim=True).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, f, 1),
                ),
                dim=1,
            )
            global_x_2 = torch.cat(
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
            global_x_2 = x

        #if True in torch.isnan(global_x):
            #print("NAN!!")
        #print("Global_", global_x.shape) #400, 384, 51
        w1 = self.attention(global_x)
        w2 = self.attention2(global_x_2)
        w = w1 + w2
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )
        
        x = torch.cat((mu, sg), 1)
        #print("22", w.shape, mu.shape, sg.shape,x.shape, w.shape) # B X 128 X 51,  B X 128, B X 128, B X 128 X 51, B X 1536 X 319 [400Frame: 64 X 1536, 64 X 1536, 64 X 1536 X 426, 64 X 1536 X 426]
        
        #print("4", x.shape) #B,256 [400Frame: B X 3072]
        x = self.bn5(x)
        #print("5", x.shape) #B X 256 [400Frame: B X 3072]
        x = self.fc6(x)
        #print("6", x.shape) #B X 512 [400Frame: B X 256]
        if self.out_bn:
            x = self.bn6(x)
        #print("7", x.shape)#BX 512 [400Frame: B X 256]
        ####################################################### Attention From RawNet ###################################################################
        '''
        if self.encoder_type == "SAP":
            x = x.permute(0,3,1,2).squeeze(-1)
            #print("1", x.shape) # 400, 51, 128
            h = torch.tanh(self.sap_linear(x))
            #print("2", h.shape, self.attention.shape) # 400, 51, 128 / 128, 1
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            #print("3", w.shape) # 400, 51
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            print(self.attention)
            #print("4", h.shape) # 400, 51, 128
            x = torch.sum(x * w, dim=1) # 400, 51, 128 / 400, 51, 128 ?????? elementwise ?????? ??????
            #print("5", x.shape) # 400, 128
        elif self.encoder_type == "ASP":
            x = x.permute(0,3,1,2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)
        '''
        #print("10", x.shape) 400, 128


        #x = x.view(x.size()[0], -1)
        #print("11", x.shape) 400, 128
        #x = self.fc(x)
        #print("12", x.shape) 400, 512

        return x


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model
