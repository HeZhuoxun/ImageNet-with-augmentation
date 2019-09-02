#!/usr/bin/env python3 -u
# This implementation is based on the Mixup-cifar10 implementation (Facebook AI)
# https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

from __future__ import print_function

import argparse
import csv
import os
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imagenet_loder import dataloder
import imagenet_models as models
from autoaugment import ImageNetPolicy
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model', default="ResNet50", type=str,
                    help='model type (default: ResNet50)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=50, type=int,
                    help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--refine_aug', default='standard', type=str,
                    help='augmentation methods: standard, clean')
parser.add_argument('--alpha', default=0.2, type=float,
                    help='mixup interpolation coefficient (default: 0.2)')
parser.add_argument('--data_dir', default='/DATA4_DB3/data/kydu/data/', type=str,
                    help='file path of ImageNet')
parser.add_argument('--no-cpu', dest='cpu', action='store_false',
                    help='dataloader with cpu(default: True)')
parser.add_argument('--procut', default=0., type=float,
                    help='the probability of cutmix when fine-tuning')
parser.add_argument('--augment', default='None', type=str,
                    help='augmentation methods: None, mixup, cutmix, autoaug')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

assert args.augment in ['None', 'mixup', 'cutmix', 'autoaug'], 'wrong augmentation method'
assert args.refine_aug in ['standard', 'clean'], 'wrong augmentation method when fine-tuning'

if args.refine_aug == 'standard':
    trainloader, train_iter, testloader, test_iter = dataloader(args.batch_size, args.data_dir, augment=True, cpu=args.cpu)
else:
    trainloader, train_iter, testloader, test_iter = dataloader(args.batch_size, args.data_dir, augment=False, cpu=args.cpu)

# if args.refine_aug == 'standard':
#     transform_train = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
# else:
#     transform_train = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])
#
# transform_test = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])
#
# traindir = os.path.join(args.data_dir, 'train')
# testdir = os.path.join(args.data_dir, 'val')
#
# trainset = datasets.ImageFolder(root=traindir, transform=transform_train)
# testset = datasets.ImageFolder(root=testdir, transform=transform_test)
#
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, num_workers=8, pin_memory=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
#                                          shuffle=False, num_workers=8)


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint/imagenet'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/imagenet/ckpt.t7' + args.model + '_' + args.name + '_'
                        + str(args.seed))
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch'] + 1
rng_state = checkpoint['rng_state']
torch.set_rng_state(rng_state)


if not os.path.isdir('results/imagenet'):
    os.mkdir('results/imagenet')
logname = ('results/imagenet/log_' + args.model + '_' + args.name + '_refine_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      nesterov=True, weight_decay=args.decay)


def mixup(x, y, alpha):
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    lam = torch.Tensor(np.random.beta(alpha, alpha, [batch_size, 1, 1, 1])
                       .astype('float32')).cuda()
    x = lam*x + (1-lam)*x[index]
    lam = lam.reshape(-1)
    return x, y, y[index], lam


def cutmix(x, y, alpha):
    batch_size, _, width, height = x.size()
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)

    c_x = np.random.randint(width)
    c_y = np.random.randint(height)
    c_w = int(width * math.sqrt(1 - lam))
    c_h = int(height * math.sqrt(1 - lam))

    x1 = np.clip(c_x - c_w // 2, 0, width)
    x2 = np.clip(c_x + c_w // 2, 0, width)
    y1 = np.clip(c_y - c_h // 2, 0, height)
    y2 = np.clip(c_y + c_h // 2, 0, height)

    x[:, :, x1:x2, y1:y2] = x[index][:, :, x1:x2, y1:y2]

    # lam = 1 - float((x2-x1)*(y2-y1)/(32*32))
    return x, y, y[index], lam


def cutout(x):
    _, _, width, height = x.size()
    lam = np.random.uniform(0, 1)

    c_x = np.random.randint(width)
    c_y = np.random.randint(height)
    c_w = int(width * math.sqrt(1 - lam))
    c_h = int(height * math.sqrt(1 - lam))

    x1 = np.clip(c_x - c_w // 2, 0, width)
    x2 = np.clip(c_x + c_w // 2, 0, width)
    y1 = np.clip(c_y - c_h // 2, 0, height)
    y2 = np.clip(c_y + c_h // 2, 0, height)

    x[:, 0, x1:x2, y1:y2] = -0.485/0.229
    x[:, 1, x1:x2, y1:y2] = -0.456/0.224
    x[:, 2, x1:x2, y1:y2] = -0.406/0.225
    return x


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()  # lam.size() == batch_size
    # return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(trainloader):
        inputs = data[0]["data"]
        targets_a = data[0]["label"].squeeze().cuda().long()
    # for batch_idx, (inputs, targets_a) in enumerate(trainloader):
    #     if use_cuda:
    #         inputs, targets_a = inputs.cuda(), targets_a.cuda()
        if args.augment == 'mixup':
            p = 0.5*(math.cos(math.pi / (30*len(trainloader)) * ((epoch - start_epoch)*len(trainloader)+batch_idx)) + 1)
            if (epoch - start_epoch) < 30 and random.random() < p:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets_a, args.alpha)
            else:
                targets_b = targets_a
                lam = torch.cuda.FloatTensor([1.]).expand(targets_a.size(0))
            outputs = net(inputs)
            loss = mixup_criterion(criterion_none, outputs, targets_a, targets_b, lam)
            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets_a.size(0)
            correct_a = predicted.eq(targets_a.data) + predicted.eq(targets_b.data)
            correct_a[correct_a == 2] = 1
            correct += correct_a.float().cpu().sum()
        elif args.augment == 'cutmix':
            if random.random() < args.procut:  # probability to cutmix
                inputs, targets_a, targets_b, lam = cutmix(inputs, targets_a, 1)
            else:
                targets_b = targets_a
                lam = 1.
            outputs = net(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets_a.size(0)
            correct_a = predicted.eq(targets_a.data) + predicted.eq(targets_b.data)
            correct_a[correct_a == 2] = 1
            correct += correct_a.float().cpu().sum()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets_a)
            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets_a.size(0)
            correct += predicted.eq(targets_a.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, train_iter,
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), 100.*correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0
    for batch_idx, data in enumerate(testloader):
    # for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = data[0]["data"]
        targets = data[0]["label"].squeeze().cuda().long()
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        test_loss += loss.data
        total += targets.size(0)
        batch_correct1, batch_correct5 = num_correct(outputs, targets, topk=(1, 5))
        correct1 += batch_correct1
        correct5 += batch_correct5

        progress_bar(batch_idx, test_iter,
                     'Loss: %.3f | Top-1 Acc: %.3f%% (%d/%d) | Top-5 Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1),
                        100.*correct1/total, correct1, total,
                        100.*correct5/total, correct5, total))
    acc1 = 100. * correct1 / total
    acc5 = 100. * correct5 / total
    if epoch == start_epoch + args.epoch - 1 or acc1 > best_acc:
        checkpoint(acc1, epoch)
    if acc1 > best_acc:
        best_acc = acc1
    return test_loss/(batch_idx+1), acc1, acc5


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint/imagenet'):
        os.mkdir('checkpoint/imagenet')
    torch.save(state, './checkpoint/imagenet/ckpt.t7' + args.model + '_' + args.name + '_refine_'
               + str(args.seed))


def num_correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k)
    return res


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 60, 120 and 180 epoch"""
    lr = args.lr
    epoch = epoch - start_epoch
    if epoch >= 25:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc',
                            'test loss', 'test top1 acc', 'test top5 acc'])

for epoch in range(start_epoch, args.epoch+start_epoch):
    # adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc1, test_acc5 = test(epoch)
    trainloader.reset()
    testloader.reset()
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), train_acc.item(), test_loss.item(),
                            test_acc1.item(), test_acc5.item()])
