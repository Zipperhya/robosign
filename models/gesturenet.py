

import torch
# import torch.nn as nn
# import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # print(type(view_shape))
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]


    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # print(group_idx.shape)
    sqrdists = square_distance(new_xyz, xyz)

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx



def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    # print(new_xyz.shape)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points



class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        # print(new_xyz.shape, new_points.shape)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        # print(new_points.shape)

        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNet_plus(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=16, radius=0.2, nsample=4, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=4, radius=0.4, nsample=2, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)


    def forward(self, x):
        B, _, _ = x.size()
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points

class get_model(nn.Module):
    def __init__(self, num_classes, bn_decay=None):
        super(get_model, self).__init__()
        self.num_classes = num_classes
        self.bn_decay = bn_decay
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        # self.lstm1 = nn.LSTM(256, 32, num_layers=2, batch_first=True, dropout=0.4, bidirectional=True)
        # self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.lstm1 = nn.LSTM(256, 32, batch_first=True, bidirectional=True, dropout=0.4)
        self.lstm2 = nn.LSTM(32 * 2, 32, batch_first=True, bidirectional=True, dropout=0.4)
        # lstm_layer = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, dropout=0.4, bidirectional=True)

        self.fc4 = nn.Linear(32 * 2, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.4)
        self.fc6 = nn.Linear(256, num_classes)


        self.point_net = PointNet_plus(in_channel=3)

    def forward(self, point_cloud):
        batch_size, num_frame, num_point, _ = point_cloud.size()
        spatial_feature_extraction_output = None
        for frame_index in range(num_frame):
            point_net_input = point_cloud[:, frame_index, :, :]

            point_net_input = point_net_input.permute(0, 2, 1)

            point_net_output = self.point_net(point_net_input)

            l3_points = point_net_output.view(batch_size, -1)
            net = F.relu(self.bn1(self.fc1(l3_points)))
            net = self.drop1(net)
            net = F.relu(self.bn2(self.fc2(net)))
            if spatial_feature_extraction_output is None:
                spatial_feature_extraction_output = net.unsqueeze(0)
            else:
                spatial_feature_extraction_output = torch.cat([spatial_feature_extraction_output, net.unsqueeze(0)], 0)
        # print(spatial_feature_extraction_output.size())
        net = spatial_feature_extraction_output.transpose(0, 1)
        # print(net.size())
        net, _ = self.lstm1(net)
        net, _ = self.lstm2(net)
        net = net[:, -1, :]

        mid_feature1 = net.detach()

        net = F.relu(self.bn4(self.fc4(net)))

        mid_feature2 = net.detach()

        net = self.drop4(net)
        net = F.relu(self.bn5(self.fc5(net)))

        mid_feature3 = net.detach()

        net = self.drop5(net)
        net = self.fc6(net)

        net = F.log_softmax(net, -1)

        return net, [mid_feature1, mid_feature2, mid_feature3]

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

import torch
import torch.nn as nn

class GestureNetBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureNetBiLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, dropout=0.4)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        print(x.shape)
        x, _ = self.lstm2(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



