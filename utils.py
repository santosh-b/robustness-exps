from __future__ import print_function
import numpy as np
import os, time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy.ndimage import filters
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
from robustness.train import train_model
from robustness.defaults import check_and_fill_args
from robustness.datasets import DataSet
from robustness.data_augmentation import TRAIN_TRANSFORMS_DEFAULT, TEST_TRANSFORMS_DEFAULT
from robustness import cifar_models
from robustness.tools import constants, helpers
from robustness import defaults, train
from cox import utils
from cox import store
import torch as ch
from argparse import ArgumentParser
from cox.utils import Parameters
import cox.store
import os
import math
import sys
from torch.autograd import Variable

class CIFAR100(DataSet):
    
    def __init__(self, data_path='/tmp/'):
        ds_kwargs = {
            'num_classes': 100,
            'mean': ch.tensor([0.5071, 0.4867, 0.4408]),
            'std': ch.tensor([0.2675, 0.2565, 0.2761]),
            'custom_class': None,
            'label_mapping': None, 
            'transform_train': TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': TEST_TRANSFORMS_DEFAULT(32)
        }
        super(CIFAR100, self).__init__('cifar100', data_path,**ds_kwargs)
    
    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        print(self.num_classes)
        return cifar_models.__dict__[arch](num_classes=self.num_classes)

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.Module):
    def __init__(self, depth, dataset='cifar10', init_weights=True, cfg=None, seed=0):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'imagenet':
            num_classes = 1000
        self.classifier = nn.Linear(cfg[-1], num_classes)
        rng = torch.manual_seed(seed)
        if init_weights:
            self._initialize_weights(rng)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        latent = x.view(x.size(0), -1)
        y = self.classifier(latent)
        if with_latent:
            return y, latent
        return y

    def _initialize_weights(self, rng):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n), generator=rng)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01, generator=rng)
                m.bias.data.zero_()

def fix_robustness_ckpt(ckpt):
    state_dict = {}
    try:
        for key in ckpt['model']:
            cleaned = key.replace('module.model.','')
            state_dict[cleaned] = ckpt['model'][key]
    except:
        for key in ckpt:
            cleaned = key.replace('module.model.','')
            state_dict[cleaned] = ckpt[key]
    return state_dict

def get_pruned_init(model, cfg, pct, dataset):
    modelnew = vgg(16, dataset=dataset, cfg=cfg)
    model.cuda()
    modelnew.cuda()

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pct)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    model = vgg(16, dataset=dataset, seed=0)
    model.cuda()

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), modelnew.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    return modelnew, cfg_mask