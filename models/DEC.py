#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    # print(idx.size())

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points

    # print(idx_base)
    if x.is_cuda:
        idx_base = idx_base.cuda()

    idx = idx + idx_base

    idx = idx.view(-1)

    # print(idx.size())
    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # print(feature.size())
    feature = feature.view(batch_size, num_points, k, num_dims)
    # print(feature.size())
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # print(x.size())
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # print(feature.size())
    return feature



class get_model(nn.Module):
    def __init__(self, output_channels, k=20, emb_dims=1024, dropout=0.5):
        super(get_model, self).__init__()
        # self.args = args
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size, frame, number,_ = x.shape
        x = x.reshape(batch_size, frame * number, 3)
        x = x.transpose(2, 1)
        x = get_graph_feature(x, k=self.k)
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x1 = x.max(dim=-1, keepdim=False)[0]
        # print(x1.size())

        x = get_graph_feature(x1, k=self.k)
        # print(x.size())
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        f = x.detach()
        # print(x.size())

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        mid_feature1 = x.detach()

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        mid_feature2 = x.detach()

        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)

        mid_feature3 = x.detach()

        x = self.dp2(x)
        x = self.linear3(x)
        x = F.log_softmax(x, -1)
        return x, [mid_feature1, mid_feature2, mid_feature3]

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


# test the get_graph_feature function
def test_get_graph_feature():
    x = torch.rand(16, 3, 1024)
    feature = get_graph_feature(x)
    # print(feature.size())

