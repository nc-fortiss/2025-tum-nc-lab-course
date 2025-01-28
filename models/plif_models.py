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

sys.path.insert(0, '/home/neumeier/Documents/corinne/libs/lava-dl/src')
import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.utils import quantize as slayer_quantize

sys.path.insert(0, '/home/neumeier/Documents/corinne/libs/spikingjelly')
from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.lava_exchange import to_lava_blocks, conv2d_to_lava_synapse_conv, to_lava_neuron, quantize_8bit, linear_to_lava_synapse_dense, to_lava_block_conv, to_lava_block_dense
from spikingjelly.activation_based.lava_exchange import BatchNorm2d as LoihiBatchNorm2d

from .adlif import LI, AdLIF



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

def quantize_8bit(weight, descale=False):
    w_scale = 1 << 6
    if descale is False:
        return slayer_quantize(
            weight, step=2 / w_scale
        ).clamp(-256 / w_scale, 255 / w_scale)
    else:
        return slayer_quantize(
            weight, step=2 / w_scale
        ).clamp(-256 / w_scale, 255 / w_scale) * w_scale

def fuse_bn(module):
    module_output = module
    if isinstance(module, (nn.Sequential,)):
        print("[nn.Sequential]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(
                module[idx + 1], nn.BatchNorm2d
            ):
                continue
            conv = module[idx]
            bn = module[idx + 1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                conv.weight
                * bn.weight[:, None, None, None]
                * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
            conv.bias.data = (
                conv.bias - bn.running_mean
            ) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()
        for name, child in module.named_children():
            module_output.add_module(name, fuse_bn(child))
        del module

    elif isinstance(module, (nn.ModuleList,)):
        print("[nn.ModuleList]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(
                module[idx + 1], nn.BatchNorm2d
            ):
                continue
            conv = module[idx]
            bn = module[idx + 1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = (
                conv.weight
                * bn.weight[:, None, None, None]
                * invstd[:, None, None, None]
            )
            if conv.bias is None:
                conv.bias = nn.Parameter(torch.zeros(conv.out_channels).to(conv.weight.device))
            conv.bias.data = (
                conv.bias - bn.running_mean
            ) * bn.weight * invstd + bn.bias
            module[idx + 1] = nn.Identity()
        module_output = module
        del module

    return module_output



class AllConvPLIFSNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, quantize=False, pretrain=False, device=None):
        super(AllConvPLIFSNN, self).__init__()

        self.v_reset = 0.0  # 0.0 # None
        self.bias = False  # False
        self.decay_inp = not quantize #False #True
        self.init_tau = 2.0  # 1 + np.abs(np.random.randn(in_channels)) # 2.0
        self.init_tau2 = 2.0  # 1 + np.abs(np.random.randn(2*in_channels)) # 2.0
        self.init_tau4 = 2.0  # 1 + np.abs(np.random.randn(4*in_channels)) # 2.0
        # neuron_params_drop = {**neuron_params}

        if quantize:
            self.quantize = quantize_8bit
        else:
            self.quantize = lambda x: x

        self.feature_extractor = torch.nn.ModuleList([
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
            layer.Dropout(p=dropout),
            layer.Flatten(),
            layer.Linear((6 * 6 * 8 * channels), feat_neur, bias=self.bias), #layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            #neuron.GatedLIFNode(T=200, surrogate_function=surrogate.ATan()),
            #layer.Linear(feat_neur, classes, bias=self.bias),
            #neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
            #                         decay_input=self.decay_inp, init_tau=self.init_tau),
        ])
        self.step_mode = 'm'
        self.pretrain = pretrain
        if pretrain:
            self.classifier = nn.Linear(feat_neur, classes, bias=self.bias)
            self.projection = nn.Sequential(nn.Linear(feat_neur, 128),
                                            nn.ReLU())
        else:
            self.classifier = nn.Sequential(layer.Linear(feat_neur, classes, bias=self.bias),
                                            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(),
                                                                     detach_reset=True, v_reset=self.v_reset,
                                                                     decay_input=self.decay_inp, init_tau=self.init_tau),
                                            )
            functional.set_step_mode(self.classifier, step_mode=self.step_mode)
            self.projection = nn.Identity()

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        functional.set_step_mode(self.feature_extractor, step_mode=self.step_mode)

        if isinstance(self.init_tau, float):
            use_cupy = False
        else:
            use_cupy = False

        if use_cupy and self.step_mode == 'm':
            functional.set_backend(self, backend='cupy')

        self.device = device
        self.lava_dl = False

    def set_step_mode(self, mode='m'):
        self.step_mode = mode
        functional.set_step_mode(self, step_mode=self.step_mode)
        if mode == 'm':
            functional.set_backend(self, backend='cupy')
        else:
            functional.set_backend(self, backend='torch')

    def forward(self, spike):
        if not self.lava_dl:
            functional.reset_net(self)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.step_mode == 'm':
            count = []
            for block in self.feature_extractor:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        else:
            print('Not implemented!')
        if self.pretrain:
            if self.lava_dl:
                rate = spike.mean(-1)
            else:
                rate = spike.mean(0)
            spike = self.classifier(rate)
            features = self.projection(rate)
        else:
            if self.lava_dl:
                features = spike.mean(-1)
            else:
                features = spike.mean(0)
            spike = self.classifier(spike)
        #spike = spike.permute(1, 2, 0).contiguous()
        return spike, features


    def get_features(self, spike):
        if not self.lava_dl:
            functional.reset_net(self)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        count = []
        for i, block in enumerate(self.feature_extractor): #[:-2]):
            #print(block)
            spike = block(spike)
            count.append(torch.mean(spike).item())
        #spike = spike.permute(1, 2, 0).contiguous()
        if self.lava_dl:
            rate = spike.mean(-1)
        else:
            rate = spike.mean(0)
        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device), rate

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
    


class AllConvPLIFLISNN(torch.nn.Module):
    def __init__(self, inp_features=2, channels=8, feat_neur=512, classes=12, delay=False, dropout=0.05, quantize=False, pretrain=False, device=None):
        super(AllConvPLIFLISNN, self).__init__()

        self.v_reset = 0.0  # 0.0 # None
        self.bias = False  # False
        self.decay_inp = not quantize  # False #True
        self.init_tau = 2.0  # 1 + np.abs(np.random.randn(in_channels)) # 2.0
        self.init_tau2 = 2.0  # 1 + np.abs(np.random.randn(2*in_channels)) # 2.0
        self.init_tau4 = 2.0  # 1 + np.abs(np.random.randn(4*in_channels)) # 2.0
        # neuron_params_drop = {**neuron_params}

        if quantize:
            self.quantize = quantize_8bit
        else:
            self.quantize = lambda x: x

        self.feature_extractor = torch.nn.ModuleList([
            layer.Conv2d(inp_features, channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(channels, 2 * channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(2 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(2 * channels, 4 * channels, kernel_size=3, padding=1, stride=2, bias=self.bias),
            layer.BatchNorm2d(4 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(4 * channels, 8 * channels, kernel_size=3, padding=0, stride=2, bias=self.bias),
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Conv2d(8 * channels, 8 * channels, kernel_size=3, padding=1, stride=1, bias=self.bias),  # stride=1
            layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
            layer.Dropout(p=dropout),
            layer.Flatten(),
            layer.Linear((6 * 6 * 8 * channels), feat_neur, bias=self.bias),
            # layer.Linear((12 * 12 * 8 * channels), feat_neur, bias=self.bias),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, v_reset=self.v_reset,
                                     decay_input=self.decay_inp, init_tau=self.init_tau),
        ])
        self.step_mode = 'm'
        self.pretrain = pretrain
        if pretrain:
            self.features = nn.Sequential(layer.Linear(feat_neur, feat_neur, bias=self.bias),
                                            LI(feat_neur, device=device))
            self.classifier = nn.Linear(feat_neur, classes, bias=self.bias)
            self.projection = nn.Sequential(nn.Linear(feat_neur, 128),
                                            nn.ReLU())
        else:
            self.classifier = nn.Sequential(layer.Linear(feat_neur, classes, bias=self.bias),
                                            LI(classes, device=device)
                                            )
            self.projection = nn.Identity()
            functional.set_step_mode(self.classifier, step_mode=self.step_mode)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,
                            (layer.BatchNorm2d, layer.BatchNorm1d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        functional.set_step_mode(self.feature_extractor, step_mode=self.step_mode)

        if isinstance(self.init_tau, float):
            use_cupy = False
        else:
            use_cupy = False

        if use_cupy and self.step_mode == 'm':
            functional.set_backend(self, backend='cupy')

        self.device = device
        self.lava_dl = False

    def set_step_mode(self, mode='m'):
        self.step_mode = mode
        functional.set_step_mode(self, step_mode=self.step_mode)
        if mode == 'm':
            functional.set_backend(self, backend='cupy')
        else:
            functional.set_backend(self, backend='torch')

    def forward(self, spike):
        if not self.lava_dl:
            functional.reset_net(self)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.step_mode == 'm':
            count = []
            for block in self.feature_extractor:
                spike = block(spike)
                count.append(torch.mean(spike).item())
        else:
            print('Not implemented!')
        if self.pretrain:
            spike = self.features(spike)
            if self.lava_dl:
                rate = spike.mean(-1)
            else:
                rate = spike.mean(0)
            spike = self.classifier(rate)
            features = self.projection(rate)
        else:
            if self.lava_dl:
                features = spike.mean(-1)
            else:
                features = spike.mean(0)
            spike = self.classifier(spike)
            spike = spike.permute(1, 2, 0).contiguous()
        return spike, features

    def get_features(self, spike):
        if not self.lava_dl:
            functional.reset_net(self)
            # print(spike.size())
            spike = spike.permute(4, 0, 1, 2, 3).contiguous()
        count = []
        for i, block in enumerate(self.feature_extractor):  # [:-2]):
            # print(block)
            spike = block(spike)
            count.append(torch.mean(spike).item())
        spike = self.features(spike)
        if self.lava_dl:
            rate = spike.mean(-1)
        else:
            rate = spike.mean(0)
        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device), rate

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
