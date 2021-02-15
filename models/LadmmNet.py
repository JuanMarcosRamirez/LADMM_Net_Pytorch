#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:58:42 2020

@author: juan
"""
# Pytorch libraries
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import time

import matplotlib.pyplot as plt


def SpectralDegradationFilter(window_size, L, q):
  kernel = torch.zeros((L//q,L,window_size,window_size))
  for i in range(0,L//q):
    kernel[i,i*q:(i+1)*(q),window_size//2,window_size//2] = 1/q
  return kernel

def ProjectionFilter(window_size, L):
  kernel = torch.zeros((1,L,window_size,window_size))
  kernel[0,1:L,window_size//2,window_size//2] = 1
  return kernel

def SpectralUpsamplingFilter(window_size, q, L):
  kernel = torch.zeros((L,L//q,window_size,window_size))
  for i in range(0,L//q):
    for j in range(0,q):
      kernel[i*q+j,i,window_size//2,window_size//2] = 1 
  return kernel

class LADMMcsifusionfastBlock(torch.nn.Module):
    def __init__(self):
        super(LADMMcsifusionfastBlock, self).__init__()

        self.lambda_step= nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr   = nn.Parameter(torch.Tensor([0.1]))
#        self.alph_upd = nn.Parameter(torch.Tensor([0.25]))
        self.rho_prmt   = nn.Parameter(torch.Tensor([1.0]))
        self.rh2_prmt   = nn.Parameter(torch.Tensor([0.1]))          

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32,  3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))

    def forward(self, x, d, r, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms):
        hs_deg  = nn.AvgPool2d(p)
        HhTHhx  = torch.mean(torch.mul(ccahs,hs_deg(x).repeat(shots_hs, 1, 1, 1)),(1))
        #HhTHhx  = F.interpolate(torch.mean(HhTHhx,(0)).repeat(L,1,1).view(1,L,M//p,N//p),scale_factor=(p,p))
        HhTHhx  = F.interpolate(torch.mean(torch.mul(HhTHhx.view(shots_hs,1,M//p,N//p).repeat(1,L,1,1), ccahs),(0)).view(1,L,M//p,N//p),scale_factor=(p,p))
    
    
        kernel  = SpectralDegradationFilter(3,L,q).cuda()
        upsamp  = SpectralUpsamplingFilter(3,q,L).cuda()
        HmTHmx  = torch.mean(torch.mul(ccams,F.conv2d(x, kernel, padding=1).repeat(shots_ms, 1, 1, 1)),(1))
        #HmTHmx  = torch.mean(HmTHmx,(0)).repeat(L,1,1).view(1,L,M,N)
        HmTHmx  = F.conv2d(torch.mean(torch.mul(HmTHmx.view(shots_ms,1,M,N).repeat(1,L//q,1,1), ccams),(0)).view(1,L//q,M,N),upsamp, padding=1)
        
        x = x - self.lambda_step * (self.rh2_prmt * x)
        x = x + self.lambda_step * (self.rh2_prmt * r)
        x = x - self.lambda_step * (HhTHhx + self.rho_prmt * HmTHmx)
        x_upd = x + self.lambda_step * (HhTyh + self.rho_prmt * HmTym)
        
        del HmTHmx, HhTHhx, HhTyh, HmTym      
        
        # Forward transform block
        x = F.conv2d(x_upd, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        
        # Soft thresholding unit
        x = x_forward + d
        x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - self.soft_thr))
        
        d = d + x_forward - x
        r = x_forward + d - x

        # Inverse transform block
        r = F.conv2d(r, self.conv1_backward, padding=1)
        r = F.relu(r)
        r = F.conv2d(r, self.conv2_backward, padding=1)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_upd - x 

        return [x_upd, d, r, symloss]


class LADMMcsifusionfastNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(LADMMcsifusionfastNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(LADMMcsifusionfastBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms):
        # mean HhTyh and HmTym
        x = 0.50 * HhTyh
        x = x + 0.50 * HmTym
        d = 0.00 * x
        r = d
        layers_sym = []
        for i in range(self.LayerNo):
            [x, d, r, layer_sym] = self.fcs[i](x, d, r, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
        
        


class ISTAcsifusionfastBlock(torch.nn.Module):
    def __init__(self):
        super(ISTAcsifusionfastBlock, self).__init__()

        self.lambda_step= nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr   = nn.Parameter(torch.Tensor([0.01]))
        self.rho_prmt   = nn.Parameter(torch.Tensor([1.0]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32,  3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))

    def forward(self, x, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms):
        
        hs_deg  = nn.AvgPool2d(p)
        HhTHhx  = torch.mean(torch.mul(ccahs,hs_deg(x).repeat(shots_hs, 1, 1, 1)),(1))
        HhTHhx  = F.interpolate(torch.mean(torch.mul(HhTHhx.view(shots_hs,1,M//p,N//p).repeat(1,L,1,1), ccahs),(0)).view(1,L,M//p,N//p),scale_factor=(p,p))
    
        kernel  = SpectralDegradationFilter(3,L,q).cuda()
        upsamp  = SpectralUpsamplingFilter(3,q,L).cuda()
        HmTHmx  = torch.mean(torch.mul(ccams,F.conv2d(x, kernel, padding=1).repeat(shots_ms, 1, 1, 1)),(1))
        HmTHmx  = F.conv2d(torch.mean(torch.mul(HmTHmx.view(shots_ms,1,M,N).repeat(1,L//q,1,1), ccams),(0)).view(1,L//q,M,N),upsamp, padding=1)
        
        x = x - self.lambda_step * (HhTHhx + self.rho_prmt * HmTHmx)
        x_input = x + self.lambda_step * (HhTyh + self.rho_prmt * HmTym)

        
        del HmTHmx, HhTHhx, HhTyh, HmTym      
        
        # Forward transform block
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        
        # Soft thresholding unit
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        
        # Inverse transform block
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_pred = F.conv2d(x, self.conv2_backward, padding=1)

#        plt.imshow(x_pred[0,31,:,:].cpu().detach().numpy())
#        plt.show()

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


class ISTAcsifusionfastNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTAcsifusionfastNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(ISTAcsifusionfastBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms):
        # mean HhTyh and HmTym
        x = 0.50 * HhTyh
        x = x + 0.50 * HmTym
        
        layers_sym = []
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, ccahs, ccams, HhTyh, HmTym, M, N, L, p, q, shots_hs, shots_ms)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
        
