# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn



class UNetLayer(nn.Module):
    def __init__(self, in_c, out_c, mode, ks, pd, activation='relu'):
        super(UNetLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.mode = mode
        self.ks = ks
        self.pd = pd
        
        if activation == 'relu':
            activation_fn = nn.ReLU(inplace=True)

        if mode == 'downsampling':
            self.down = nn.Sequential(
                nn.AvgPool2d(ks, stride=2, padding=pd)
                )            
            self.l1 = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=1, padding=pd),
                activation_fn,
            )

        elif mode == 'upsampling' or 'upsampling_noskip':
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
            self.l1 = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=1, padding=pd),
                activation_fn,
            )
        elif mode == 'regular':
            self.l1 = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=ks, stride=1, padding=pd),
                activation_fn,
            )

    def forward(self, x, skip_x=None):
        if self.mode == 'downsampling':
            return self.l1(self.down(x))
        elif self.mode == 'upsampling':
            return self.l1(torch.cat([self.up(x), skip_x], 1))
        elif self.mode == 'upsampling_noskip':
            return self.l1(self.up(x))
        elif self.mode == 'regular':
            return self.l1(x)




class UNet2d(nn.Module):
    def __init__(self, in_c, out_c, nf=32, sr=1):
        super(UNet2d, self).__init__()

        self.layer0 = UNetLayer(in_c, nf, 'regular', 3, 1)
        self.layer1 = UNetLayer(nf, nf, 'regular', 3, 1)
        self.layer2 = UNetLayer(nf, nf*2, 'downsampling', 3, 1)
        self.layer3 = UNetLayer(nf*2, nf*2, 'regular', 3, 1)
        self.layer4 = UNetLayer(nf*2, nf*4, 'downsampling', 3, 1)
        self.layer5 = UNetLayer(nf*4, nf*4, 'regular', 3, 1)        
        self.layer6 = UNetLayer(nf*6, nf*2, 'upsampling', 3, 1)         
        self.layer7 = UNetLayer(nf*2, nf*2, 'regular', 3, 1)
        self.layer8 = UNetLayer(nf*3, nf, 'upsampling', 3, 1)        
        self.layer9 = UNetLayer(nf, nf, 'regular', 3, 1)
        self.out_normal = nn.Conv2d(nf, out_c, kernel_size=3, stride=1, padding=1)         
        self.tanh = nn.Tanh()        
        

    def forward(self, x):

        assert(torch.equal(torch.max(torch.isnan(x)), torch.tensor(False).cuda()))

        x = 2.*x - 1.0 # convert input values from [0, 1] to [-1, 1]

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)        
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5, layer3)
        layer7 = self.layer7(layer6)
        layer8 = self.layer8(layer7, layer1)
        layer9 = self.layer9(layer8)
        out = self.out_normal(layer9)
 
        return (self.tanh(out) + 1.0) / 2.0



class NPTnet(nn.Module):
    def __init__(self):
        super(NPTnet, self).__init__()
        self.net = UNet2d(in_c=6, out_c=3, nf=16)
        
    def forward(self, src_left, src_right):
        net_in = (torch.cat((src_left, src_right),dim=1)).float()
        net_out = self.net(net_in)
        return net_out

