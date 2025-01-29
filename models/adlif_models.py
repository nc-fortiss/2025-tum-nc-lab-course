# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import lava.lib.dl.slayer as slayer
import sys


from spikingjelly.activation_based import surrogate, neuron, functional, layer

from .adlif import AdLIF

def calc_feature_loss(rates, targets):
    loss = nn.CosineEmbeddingLoss().to(rates.device)
    batch_size = targets.shape[0]
    feat1, feat2 = torch.split(rates, split_size_or_sections=batch_size // 2, dim=0)
    target1, target2 = torch.split(targets, split_size_or_sections=batch_size // 2, dim=0)
    y = 2*torch.eq(target1, target2).float() - 1.0
    error = loss(feat1, feat2, y)
    #print(error)
    return error

def ann_classifier(y):
    pred = torch.argmax(y, dim=1)
    return pred


    
class AllConvPLIFAdLIFSNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, quantize=False, device=None):
        super(AllConvPLIFAdLIFSNN, self).__init__()

        self.v_reset = 0.0  # 0.0 # None
        self.bias = False  # False
        self.decay_inp = not quantize #False #True
        self.init_tau = 2.0  # 1 + np.abs(np.random.randn(in_channels)) # 2.0
        self.init_tau2 = 2.0  # 1 + np.abs(np.random.randn(2*in_channels)) # 2.0
        self.init_tau4 = 2.0  # 1 + np.abs(np.random.randn(4*in_channels)) # 2.0
        # neuron_params_drop = {**neuron_params}


        self.blocks = torch.nn.ModuleList([
            layer.Conv2d(inp_features, channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(channels, 2*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(2*channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(4*channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(4*channels, 8 * channels, kernel_size=3, padding=0, stride=2, bias=self.bias),
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(8 * channels, 8 * channels, kernel_size=3, padding=1, stride=1, bias=self.bias), # stride=1
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            #AdLIF(8 * channels, 8 * channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Flatten(),
            layer.Linear((6 * 6 * 8 * channels), feat_neur, bias=self.bias), #layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            AdLIF((6 * 6 * 8 * channels), feat_neur, threshold=1.0, rnn=False, device=device),
        ])
        self.out_linear = nn.Linear(feat_neur, classes, bias=self.bias)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.step_mode = 'm'
        functional.set_step_mode(self.blocks, step_mode=self.step_mode)
        # functional.set_step_mode(self.conv, step_mode=self.step_mode)

        if isinstance(self.init_tau, float):
            use_cupy = False
        else:
            use_cupy = False

        if use_cupy and self.step_mode == 'm':
            functional.set_backend(self, backend='cupy')

        self.device = device

    def forward(self, spike):
        functional.reset_net(self.blocks)
        # print(spike.size())
        spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.step_mode == 'm':
            count = []
            for block in self.blocks:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        else:
            print('Not implemented!')
        spike = self.out_linear(spike.mean(0))
        #spike = spike.permute(1, 2, 0).contiguous()
        return spike, torch.FloatTensor(count).reshape(
            (1, -1)
        ).to(spike.device)


    def get_features(self, spike):
        functional.reset_net(self.blocks)
        # print(spike.size())
        spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        count = []
        for i, block in enumerate(self.blocks): #[:-2]):
            #print(block)
            spike = block(spike)
            count.append(torch.mean(spike).item())
        rate = spike.mean(0)
        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device), rate

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = []
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear)):
                if m.weight.grad is None:
                    grad.append(0)
                else:
                    grad.append(torch.norm(m.weight.grad).item() / torch.numel(m.weight.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename, add_input_layer=False, input_dims=[2, 128, 128]):
        # network export to hdf5 format
        return 0
    
class AllConvAdLIFLISNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, ds=6, quantize=False, device=None):
        super(AllConvAdLIFLISNN, self).__init__()

        self.v_reset = 0.0  # 0.0 # None
        self.bias = False  # False
        self.decay_inp = not quantize #False #True
        self.init_tau = 2.0  # 1 + np.abs(np.random.randn(in_channels)) # 2.0
        self.init_tau2 = 2.0  # 1 + np.abs(np.random.randn(2*in_channels)) # 2.0
        self.init_tau4 = 2.0  # 1 + np.abs(np.random.randn(4*in_channels)) # 2.0
        # neuron_params_drop = {**neuron_params}


        self.blocks = torch.nn.ModuleList([
            layer.Conv2d(inp_features, channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            AdLIF(channels, channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Conv2d(channels, 2*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(2*channels, momentum=0.01, eps=1e-3),
            AdLIF(2*channels, 2*channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Conv2d(2*channels, 4*channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(4*channels, momentum=0.01, eps=1e-3),
            AdLIF(4*channels, 4*channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Conv2d(4*channels, 8 * channels, kernel_size=3, padding=0, stride=2, bias=self.bias),
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            AdLIF(8*channels, 8*channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Conv2d(8 * channels, 8 * channels, kernel_size=3, padding=1, stride=1, bias=self.bias), # stride=1
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            AdLIF(8*channels, 8*channels, threshold=1.0, rnn=False, device=device),
            layer.Dropout(p=dropout),
            layer.Flatten(),
            layer.Linear((6 * 6 * 8 * channels), feat_neur, bias=self.bias), #layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            AdLIF((6 * 6 * 8 * channels), feat_neur, threshold=1.0, rnn=False, device=device),
            layer.Linear(feat_neur, feat_neur, bias=self.bias), #layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            LI(feat_neur, device=device),
            #neuron.GatedLIFNode(T=200, surrogate_function=surrogate.ATan()),
            #layer.Linear(feat_neur, classes, bias=self.bias),
            #neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
            #                         decay_input=self.decay_inp, init_tau=self.init_tau),
        ])
        self.out_linear = nn.Linear(feat_neur, classes, bias=self.bias)
        #self.projection = nn.Sequential(nn.Linear(feat_neur, 128),
        #                                nn.ReLU())
        self.projection = nn.Identity()

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.step_mode = 'm'
        functional.set_step_mode(self.blocks, step_mode=self.step_mode)
        # functional.set_step_mode(self.conv, step_mode=self.step_mode)

        if isinstance(self.init_tau, float):
            use_cupy = True  # False
        else:
            use_cupy = False

        if use_cupy and self.step_mode == 'm':
            functional.set_backend(self, backend='cupy')

        self.device = device
        self.lava_dl = False

    def set_step_mode(self, mode='m'):
        self.step_mode = mode
        functional.set_step_mode(self.blocks, step_mode=self.step_mode)
        if mode == 'm':
            functional.set_backend(self, backend='cupy')
        else:
            functional.set_backend(self, backend='torch')

    def forward(self, spike):
        if not self.lava_dl:
            functional.reset_net(self.blocks)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.step_mode == 'm':
            count = []
            for block in self.blocks:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        else:
            print('Not implemented!')
        #if self.lava_dl:
        #    rate = spike.mean(-1)
        #else:
        #    rate = spike.mean(0)
        # last
        #spike = spike[-1]
        # mean
        spike = spike.mean(0)
        out = self.out_linear(spike)
        features = self.projection(spike)
        #spike = spike.permute(1, 2, 0).contiguous()
        return out, features
    

    def get_features(self, spike):
        if not self.lava_dl:
            functional.reset_net(self.blocks)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        count = []
        if self.step_mode == 'm':
            for block in self.blocks:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        elif self.step_mode == 's': #single-step inference mode
            out = []
            for t in range(spike.shape[-1]):
                spike = spike[t, ...]
                for block in self.blocks:
                    spike = block(spike)
                    count.append(torch.mean(spike).item())
                out.append(spike)
            out = torch.stack(out, dim=0)
        #if self.lava_dl:
        #    rate = spike.mean(-1)
        #else:
        #    rate = spike.mean(0)
        # last
        #spike = spike[-1]
        # mean
        spike = spike.mean(0)
        features = self.projection(spike)
        return spike, features, spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = []
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, layer.Linear, nn.Conv3d)):
                if m.weight.grad is None:
                    grad.append(0)
                else:
                    grad.append(torch.norm(m.weight.grad).item() / torch.numel(m.weight.grad))

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
