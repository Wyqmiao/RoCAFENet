import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor

import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x
        
class BasicConv2dReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def global_median_pooling(x):  

    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled 


class MCA(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(MCA, self).__init__()
      
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        
        avg_out = self.fc1(avg_pool)
        avg_out = F.relu(avg_out, inplace=True) 
        avg_out = self.fc2(avg_out)
        avg_out = torch.sigmoid(avg_out)

       
        max_out = self.fc1(max_pool)
        max_out = F.relu(max_out, inplace=True) 
        max_out = self.fc2(max_out) 
        max_out = torch.sigmoid(max_out) 

        
        median_out = self.fc1(median_pool) 
        median_out = F.relu(median_out, inplace=True) 
        median_out = self.fc2(median_out) 
        median_out = torch.sigmoid(median_out) 

        
        out = avg_out + max_out + median_out
        return out


class CSDPM(nn.Module):
    def __init__(self, channel, channel_attention_reduce=4):
        super(CSDPM , self).__init__()

        self.C = channel
        self.O = channel
        
        assert channel == channel, "Input and output channels must be the same"
        
        self.channel_attention = MCA(input_channels=channel,
                                                  internal_neurons=channel // channel_attention_reduce)

       
        self.initial_depth_conv = nn.Conv2d(channel, channel, kernel_size=5, padding=2, groups=channel)

        
        self.depth_convs = nn.ModuleList([

            nn.Conv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(1, 11), padding=(0, 5), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(11, 1), padding=(5, 0), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(1, 21), padding=(0, 10), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(21, 1), padding=(10, 0), groups=channel),
        ])

        self.pointwise_conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
    
        inputs = self.pointwise_conv(inputs) 
        inputs = self.act(inputs)

        channel_att_vec = self.channel_attention(inputs)
        inputs = channel_att_vec * inputs

        initial_out = self.initial_depth_conv(inputs)

        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out

 
class DiagonalLowerSplit(nn.Module):
    def __init__(self):
        super(DiagonalLowerSplit, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        mask = torch.triu(torch.ones(height, width), diagonal=1).to(x.device)
        
        x = x * (1 - mask.unsqueeze(0).unsqueeze(0))  
        
        return x

class DiagonalUpperSplit(nn.Module):
    def __init__(self):
        super(DiagonalUpperSplit, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        mask = torch.tril(torch.ones(height, width), diagonal=0).to(x.device)
        
        x = x * mask.unsqueeze(0).unsqueeze(0)  
        
        return x

class DIRB(nn.Module): 
    def __init__(self):
        super(DIRB, self).__init__()
        self.diagonal_lower_split = DiagonalLowerSplit()
        self.diagonal_upper_split = DiagonalUpperSplit()

    def rotate_tensor_90(self, tensor):
        
        return tensor.rot90(1, dims=(-2, -1))

    def forward(self, x):    
       
        x1 = self.diagonal_lower_split(x)
        x2 = self.diagonal_upper_split(x)
        
        x_rotated = self.rotate_tensor_90(x)
        
        x3 = self.diagonal_lower_split(x_rotated)
        x4 = self.diagonal_upper_split(x_rotated)
        
        x_Diagonal1 = x1 + x4
        x_Diagonal2 = x2 + x3
        
        return x_Diagonal1, x_Diagonal2



 
class SGCB(nn.Module):  #(Spatial Grouped Coordinate Attention) 空间分组坐标注意力
    def __init__(self, channel, h, w, reduction = 16, num_groups = 4):
        super(SGCB, self).__init__()
        self.num_groups = num_groups  
        self.group_channels = channel // num_groups  
        self.h = h 
        self.w = w  

        
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))  # 输出大小为(h, 1)
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
       
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))  # 输出大小为(1, w)
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))



        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((h, w))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((h, w))

       
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1))
        )
        
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()


    def forward(self, x):
        batch_size, channel, height, width = x.size()
       
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        x = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        
        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)

       
        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)

        
        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, self.h, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, self.h, 1))

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, self.w))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, self.w))

        
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, self.h, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, self.w)

     
        out_all = x * att_h * att_w 
        out_all = out_all.view(batch_size, channel, height, width)

        return out_all




class SDSM(nn.Module):
    def __init__(self, channel, h, w):
        super(SDSM, self).__init__()
        self.diagonal = DIRB()       
        self.SGCB = SGCB(channel, h, w)
        
        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)

        self.gamma_cur = nn.Parameter(torch.ones(1))
        
        self.conv = nn.Sequential(
            BasicConv2dReLu(channel, channel, 3, padding=1),
            nn.Dropout2d(0.1, False),
            BasicConv2dReLu(channel, channel, 1)
        )

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B X C X H X W)
        returns:
            out : attention value + input feature
            attention: C X H x W
        """
        x1, x2 = self.diagonal(x)
        
        proj_query = self.SGCB(x1)
        proj_key = self.SGCB(x2)
        proj_value = x

        
        query = self.query_conv(proj_query)
        key = self.key_conv(proj_key)
        value = self.value_conv(proj_value)
        
        
        B, C, H, W = query.size()
        query = query.view(B, C, -1)
        key = key.view(B, C, -1)
        value = value.view(B, C, -1)
        
        
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)
        attention_scores = self.softmax(attention_scores)

        attention_output = torch.bmm(value, attention_scores)
        attention_output = attention_output.view(B, C, H, W)
        out = self.gamma_cur * self.conv(attention_output) + x
        
        return out


        
class AFFH(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(AFFH, self).__init__()

        
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)), 
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )

        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(in_channels)  
        )

    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

    
    def forward(self, x):
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  
        x = x * x_channel_att  

        x = self.channel_shuffle(x, groups=4) 

        x_spatial_att = self.spatial_attention(x).sigmoid()  

        out = x * x_spatial_att  
       
        return out  


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        
        self.AFFH1 = AFFH(32)
        self.AFFH2 = AFFH(96)
        self.AFFH3 = AFFH(96)
        self.AFFH4 = AFFH(96)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 32*11*11
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(32, 32, 3, padding=1)  
        )
               

        # 32*22*22
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(96, 32, 3, padding=1) 
        )
        
        # 32*44*44
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(96, 32, 3, padding=1)  
        )
        
        # 32*88*88
        self.decoder1 = nn.Sequential(
            BasicConv2d(96, 32, 3, padding=1)  
        )
        
        self.conv = nn.Conv2d(channel, 1, 1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        

    def forward(self, x4, x3, x2, x1):

        x4_affh = self.AFFH1(x4)
        x4_decoder = self.decoder4(x4_gcsa)  # 32*22*22
        x4 = self.upsample(x4) #  32*22*22

        x3_cat = torch.cat([x4_decoder, x3, x4], 1)   # 96*22*22
        x3_affh = self.AFFH2(x3_cat)        
        x3_decoder = self.decoder3(x3_gcsa)
        x3 = self.upsample(x3) #  32*44*44

        x2_cat = torch.cat([x3_decoder, x2, x3], 1) # 96*44*44
        x2_affh = self.AFFH3(x2_cat)        
        x2_decoder = self.decoder2(x2_gcsa)   # 32*88*88  
        x2 = self.upsample(x2) #  32*88*88       

        x1_cat = torch.cat([x2_decoder, x1, x2], 1) # 96*88*88
        x1_affh = self.AFFH4(x1_cat)        
        x1_decoder = self.decoder1(x1_gcsa)   # 32*88*88

        
        x = self.conv(x1_decoder) # 1*88*88
        x = self.upsample_4(x) # 1*352*352

        return x
        
        

class RoCAFENet(nn.Module):
    def __init__(self, channel=32):
        super(RoCAFENet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 1, 1) # 128x44x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        
        # 模块1

        self.CSDPM1 = CSDPM(64, channel_attention_reduce=4)
        self.CSDPM2 = CSDPM(128, channel_attention_reduce=4)
        self.CSDPM3 = CSDPM(320, channel_attention_reduce=4)
        self.CSDPM4 = CSDPM(512, channel_attention_reduce=4)
 
         
        # 模块2
        self.SDSM1 = SDSM(64, h=88, w=88)
        self.SDSM2 = SDSM(128, h=44, w=44)
        self.SDSM3 = SDSM(320, h=22, w=22)
        self.SDSM4 = SDSM(512, h=11, w=11)
        

        
        self.Decoder = Decoder(channel)
        
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11



        x1_csdpm = self.CSDPM1(x1)
        x2_csdpm = self.CSDPM2(x2)
        x3_csdpm = self.CSDPM3(x3)
        x4_csdpm = self.CSDPM4(x4)

 
        x1_sdsm = self.SDSM1(x1)
        x2_sdsm = self.SDSM2(x2)
        x3_sdsm = self.SDSM3(x3)
        x4_sdsm = self.SDSM4(x4)
       
        x1_all = x1_sdsm + x1_csdpm
        x2_all = x2_sdsm + x2_csdpm
        x3_all = x3_sdsm + x3_csdpm
        x4_all = x4_sdsm + x4_csdpm
        
        x1_nor = self.ChannelNormalization_1(x1_all) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2_all) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3_all) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_all) # 32x11x11
        
        prediction = self.Decoder(x4_nor, x3_nor, x2_nor, x1_nor)


        return prediction, self.sigmoid(prediction)
