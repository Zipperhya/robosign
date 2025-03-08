"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
# import util as provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
# from dataloader import RadarDataset
from robot_dataloader import RadarDataset_offline as RobotRadarDataset
from human_dataloader import RadarDataset_offline as HumanRadarDataset
from torch.utils.data import random_split
# from data_utils import get_dataset
from utils import Metrics

from models.DAN import ForeverDataIterator, MultipleKernelMaximumMeanDiscrepancy, GaussianKernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=10, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=50, help='Point Number')
    parser.add_argument('--num_frame', type=int, default=1, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    parser.add_argument('--data_path_offline', help='data path from offline')
    # parser.add_argument('--')
    parser.add_argument('--train_size', type=float, default=0.8, help='train size')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size')

    ''' Adaptation Parameters '''
    parser.add_argument('-i', '--iters-per-epoch', default=20, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--overall_epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--non-linear', default=False, action='store_true',
                        help='whether not use the linear version')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--print_freq', default=25, type=int,
                        help='the frequency of print information')
    parser.add_argument('--target_data_path_offline', help='target data path from offline')

    parser.add_argument('--pretrained', default=True, action='store_true', help='use pretrained model')

    parser.add_argument('--pretrained_path', help='pretrained model path')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(classifier, loader, loader_target, mkmmd_loss, criterion, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = classifier.eval()
    mkmmd_loss = mkmmd_loss.eval()

    targets = []
    preds = []
    mean_loss = []

    source_iter = ForeverDataIterator(loader)
    target_iter = ForeverDataIterator(loader_target)

    for i in tqdm(range(args.iters_per_epoch)):

        points_s, target_s = next(source_iter)[:2]
        points_t, target_t = next(target_iter)[:2]

        points_s = points_s.clone().detach().float()
        points_t = points_t.clone().detach().float()

        target_s = target_s.clone().detach().long()

        if not args.use_cpu:
            points_s, target_s, points_t = points_s.cuda(), target_s.cuda(), points_t.cuda()

        pred_s, feat_s = classifier(points_s)
        pred_t, feat_t = classifier(points_t)
        # print(feat_s)
        # print(feat_t.size())

        cls_loss = criterion(pred_s, target_s.long(), feat_s)
        # loss = cls_loss
        # transfer_loss = 0.0
        for idx in range(len(feat_s)):
            cls_loss += mkmmd_loss(feat_s[idx], feat_t[idx]) * args.trade_off
        # loss = cls_loss + transfer_loss * args.trade_off
        loss = cls_loss

        # points = points.transpose(2, 1)
        # pred, _ = classifier(points)
        pred_choice = pred_s.data.max(1)[1]

        preds.append(pred_choice.cpu().numpy())
        targets.append(target_s.cpu().numpy())

        # print(pred_choice.cpu().numpy())
        # print(target.cpu().numpy())
        # print(np.unique(target.cpu().numpy()))
        for cat in np.unique(target_s.cpu()):
            classacc = pred_choice[target_s == cat].eq(target_s[target_s == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points_s[target_s == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target_s.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points_s.size()[0]))
        mean_loss.append(loss.item() / float(points_s.size()[0]))
    # print(class_acc[:, 0])
    # print(class_acc[:, 1])

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    class_loss = np.mean(mean_loss)

    return instance_acc, class_acc, class_loss


def train(source_loader, target_loader, classifier, criterion,
          mkmmd_loss, optimizer, lr_scheduler, global_epoch,
          epoch, args, logger):

    def log_string(str):
        logger.info(str)
        print(str)

    source_iter = ForeverDataIterator(source_loader)
    target_iter = ForeverDataIterator(target_loader)

    log_string('Epoch %d (%d/%s): start' % (global_epoch + 1, epoch + 1, args.epoch))
    mean_correct = []
    mean_acc_loss = []
    mean_mmd_loss = []

    classifier = classifier.train()
    mkmmd_loss = mkmmd_loss.train()

    lr_scheduler.step()
    # log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
    # mean_correct = []
    # classifier = classifier.train()

    # scheduler.step()
    for i in tqdm(range(args.iters_per_epoch)):
        optimizer.zero_grad()


        points_s, target_s = next(source_iter)[:2]
        points_t, target_t = next(target_iter)[:2]

        points_s = points_s.clone().detach().float()
        points_t = points_t.clone().detach().float()

        target_s = target_s.clone().detach().long()


        if not args.use_cpu:
            points_s, target_s, points_t = points_s.cuda(), target_s.cuda(), points_t.cuda()

        pred_s, feat_s = classifier(points_s)
        pred_t, feat_t = classifier(points_t)
        # print(feat_s)
        # print(feat_t.size())

        cls_loss = criterion(pred_s, target_s.long(), feat_s)
        # loss = cls_loss
        # transfer_loss = 0.0
        for idx in range(len(feat_s)):
            cls_loss += mkmmd_loss(feat_s[idx], feat_t[idx]) * args.trade_off
        # loss = cls_loss + transfer_loss * args.trade_off
        loss = cls_loss
        pred_choice = pred_s.data.max(1)[1]

        correct = pred_choice.eq(target_s.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points_s.size()[0]))
        mean_acc_loss.append(cls_loss.item() / float(points_s.size()[0]))
        # mean_mmd_loss.append(transfer_loss.item() / float(points_s.size()[0]))
        loss.backward()
        optimizer.step()
        # global_step += 1

    train_instance_acc = np.mean(mean_correct)
    log_string('Train Instance Accuracy: %f' % train_instance_acc)
    train_instance_clf_loss = np.mean(mean_acc_loss)
    # train_instance_mmd_loss = np.mean(mean_mmd_loss)
    log_string('Train CLF Loss: %f' % train_instance_clf_loss)
    # log_string('Train MMD Loss: %f' % train_instance_mmd_loss)
    return train_instance_acc, train_instance_clf_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    # exp_dir = exp_dir.joinpath('pretest')
    # exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # online dataset load
    # full_dataset = RadarDataset(directory= args.data_path, N=args.num_point, F=args.num_frame)

    # offline dataset load
    full_dataset = RobotRadarDataset(data_path=args.data_path_offline, N=args.num_point, F=args.num_frame)

    train_size = int(args.train_size * len(full_dataset))  # 80% for training
    test_size = int(args.test_size * len(full_dataset))  # 20% for testing
    remain_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, remain_dataset = random_split(full_dataset, [train_size, test_size, remain_size],
                                                               generator=torch.Generator().manual_seed(42))

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


    full_dataset_target = HumanRadarDataset(data_path=args.target_data_path_offline, N=args.num_point, F=args.num_frame)
    train_size_target = int(0.8 * len(full_dataset_target))  # 80% for training
    test_size_target = len(full_dataset_target) - train_size_target  # 20% for testing
    train_dataset_target, test_dataset_target = random_split(full_dataset_target, [train_size_target, test_size_target], generator=torch.Generator().manual_seed(42))
    trainDataLoader_target = torch.utils.data.DataLoader(train_dataset_target, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader_target = torch.utils.data.DataLoader(test_dataset_target, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)



    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('./robot_dataloader.py', str(exp_dir))
    shutil.copy('./train_robot_adaptation.py', str(exp_dir))

    classifier = model.get_model(num_class)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    # adaptation loss:
    mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=not args.non_linear
    )

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        mkmmd_loss = mkmmd_loss.cuda()

    if args.pretrained:
        checkpoint = torch.load(args.pretrained_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        start_epoch = 0

    else:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_loss = 1e10

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):

        train_acc, train_loss = train(trainDataLoader, trainDataLoader_target, classifier, criterion,
          mkmmd_loss, optimizer, scheduler, global_epoch,
          epoch, args, logger)



        with torch.no_grad():
            instance_acc, class_acc, class_loss = test(classifier.eval(), testDataLoader, testDataLoader_target, mkmmd_loss.eval(),criterion.eval(), num_class=num_class)

            # metrics = Metrics(targets, preds, num_class, './robot_plot')

            if class_loss < best_loss:
                best_loss = class_loss
                log_string('Best loss: %f' % best_loss)
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            #

        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)