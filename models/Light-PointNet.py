import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=kernel_size),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class Linear_Block(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.linear_block = nn.Sequential(
            nn.Linear(in_features=self.in_feature, out_features=self.out_feature),
            nn.BatchNorm1d(self.out_feature),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_block(x)


class Linear_Classifier(nn.Module):
    def __init__(self, in_feature, out_put):
        super().__init__()


        self.linear1 = nn.Sequential(
            nn.Linear(in_feature, 128),
            nn.ReLU()
            # nn.Softmax(dim=1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(128, out_put),
            # nn.ReLU()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear2(self.linear1(x))

        return x


class Transformation_Net(nn.Module):

    def __init__(self, in_channel,
                 out_channels,
                 out_features,
                 kernel_size=1,
                 padding=1,
                 stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.out_features = out_features

        self.conv_block1 = Conv_Block(self.in_channel, self.out_channels[0], kernel_size)
        self.conv_block2 = Conv_Block(self.out_channels[0], self.out_channels[1], kernel_size)
        self.conv_block3 = Conv_Block(self.out_channels[1], self.out_channels[2], kernel_size)

        # self.max_pooling1 = nn.MaxPool1d(kernel_size=AS)

        self.liner_block1 = Linear_Block(in_feature=self.out_channels[-1], out_feature=self.out_features[0])
        self.liner_block2 = Linear_Block(in_feature=self.out_features[0], out_feature=self.out_features[1])
        self.liner_block3 = Linear_Block(in_feature=self.out_features[1], out_feature=self.out_features[2])

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, in_channel * in_channel)
        self.relu = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        input_x = x.detach()

        x = x.transpose(1, 2)

        # print(x.shape)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # x = self.max_pooling1(x)

        x = torch.max(x, dim=2, keepdim=True)[0]

        x = x.squeeze()

        # x = self.liner_block1(x)
        # x = self.liner_block2(x)
        # x = self.liner_block3(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape(x.shape[0], self.in_channel, x.shape[1] // self.in_channel)

        # x = torch.matmul(input_x, x)
        x = torch.bmm(input_x, x)

        return x


class point_cloud_global_embedding(nn.Module):

    def __init__(self, in_channel=3):
        super().__init__()

        self.Transformation_Net1 = Transformation_Net(in_channel, [64, 128, 1024], [512, 256, 9])
        self.Transformation_Net2 = Transformation_Net(64, [64, 128, 1024], [512, 256, 4096])

        self.stn = STN3d(in_channel)


        self.conv_block1 = Conv_Block(3, 64)
        # self.conv_block2 = Conv_Block(64, 64)

        # self.conv_block3 = Conv_Block(64, 64)
        self.conv_block4 = Conv_Block(64, 128)
        self.conv_block5 = Conv_Block(128, 1024)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.max_pooling1 = nn.MaxPool1d(kernel_size=AS)

    def forward(self, x):
        # x = x.transpose(1, 2)

        x = self.Transformation_Net1(x)

        # x = self.stn(x)
        # x = x.transpose(2, 1)
        # trans_feat = self.stn(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans_feat)
        # x = x.transpose(2, 1)

        # print(x.shape)

        x = x.transpose(1, 2)

        x = self.conv_block1(x)
        # x = self.conv_block2(x)

        x = x.transpose(1, 2)

        # print(x.shape)

        x = self.Transformation_Net2(x)

        x = x.transpose(1, 2)

        # print(x.shape)

        # x = self.conv_block3(x)
        x = self.conv_block4(x)
        # x = self.conv_block5(x)
        x = self.bn3(self.conv3(x))
        # x = self.max_pooling1(x)

        x = torch.max(x, dim=2, keepdim=True)[0]

        x = x.squeeze()

        return x


class Temporal_Net(nn.Module):
    def __init__(self, in_channel, output):
        super().__init__()

        self.lstm = nn.GRU(in_channel, 256, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, output),
            # nn.Softmax(dim=1)
        )

        # self.lin0 = nn.Linear(256, 128)
        # self.lin1 = nn.Linear(10, output)
        # self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # x, _ = self.lstm(x)

        # x = x[:,-1,:]
        # x = self.drop(x)
        x = self.classifier(x)
        return x


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)

        # print(trans.shape)

        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)

        # print(x.size())

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        # print(pointfeat.size())
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]


        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class get_model(nn.Module):
    def __init__(self,
                 out_size,
                 in_channels=3,
                 embedding_len=1024,):
        super().__init__()

        self.embedding_model = point_cloud_global_embedding(in_channels)
        # self.pointnet = PointNetEncoder(channel=in_channels)

        self.loss = nn.CrossEntropyLoss()
        self.out_size = out_size

        # BiliLSTM + Dropout
        self.lstm = nn.LSTM(embedding_len, 256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        self.classifier_net = Linear_Classifier(256, out_size)

        # self.lr = lr
        # self.optimizer_name = optimizer_name

    def forward(self, x):

        '''
        x: (B, F, N, 3)
        '''

        batch_size = x.shape[0]
        frame = x.shape[1]

        x = x.reshape((batch_size * frame, x.shape[2], x.shape[3]))


        # pointnet
        # x = x.transpose(1, 2)
        # x = self.pointnet(x)

        # embedding model
        x = self.embedding_model(x)

        x = x.reshape((batch_size, frame, -1))

        # mid_feature1 = x.detach()

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        mid_feature2 = x.detach()

        x = self.classifier_net(x)

        x = F.log_softmax(x, -1)

        return x, [mid_feature2]

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
