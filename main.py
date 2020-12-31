#!coding:utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from util import datasets, Trainer
from architectures.fpn import Fpn_n
import itertools

from util.datasets import NO_LABEL


def data_loaders(
        train_transform,
        eval_transform,
        heat_transform,
        datadir,
        batch_size,
        batch_size_ul,
        config):
    evaldir = os.path.join(datadir, config.eval_subdir)
    evalset = datasets.dataloader(root=evaldir,
                                  list_path=config.test_list_path,
                                  transform=eval_transform,
                                  target_transform=heat_transform,
                                  max_iters=None)
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=batch_size * 4,
                                              shuffle=False,
                                              num_workers=2 * config.workers,
                                              pin_memory=True,
                                              drop_last=False)  # 测试用的
    traindir = os.path.join(datadir, config.train_subdir)
    if config.list_path2 != 'None':
        trainset_l = datasets.dataloader(root=traindir,
                                         list_path=config.list_path2,
                                         transform=train_transform,
                                         target_transform=heat_transform,
                                         max_iters=None)  # 这里缺最大数据量
        labeled_idxs = list(range(len(trainset_l.sample_files)))
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
        train_l_loader = torch.utils.data.DataLoader(trainset_l,
                                                     batch_sampler=batch_sampler,
                                                     num_workers=config.workers,
                                                     pin_memory=True)
    if config.list_path1 != 'None':
        trainset_ul = datasets.dataloader(root=traindir,
                                          list_path=config.list_path1,
                                          transform=train_transform,
                                          target_transform=heat_transform,
                                          max_iters=None)  # 这里缺最大数据量
        unlabeled_idxs = list(range(len(trainset_ul.sample_files)))
        sampler = SubsetRandomSampler(unlabeled_idxs)
        batch_sampler = BatchSampler(sampler, batch_size_ul, drop_last=False)
        train_ul_loader = torch.utils.data.DataLoader(trainset_ul,
                                                      batch_sampler=batch_sampler,
                                                      num_workers=config.workers,
                                                      pin_memory=True)
    if config.list_path2 != 'None' and config.list_path1 != 'None':
        # train_loader = itertools.zip_longest(train_l_loader, train_ul_loader)
        train_loader = {'l': train_l_loader, 'ul': train_ul_loader}
    elif config.list_path2 == 'None' and config.list_path1 != 'None':
        train_loader = train_ul_loader
    elif config.list_path1 == 'None' and config.list_path2 != 'None':
        train_loader = train_l_loader

    return train_loader, eval_loader


def create_loss_fn(config):
    if config.loss == 'mse':
        # for pytorch 0.4.0
        # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction=None)
        criterion = nn.MSELoss()
        # for pytorch 0.4.1
        # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none')
    return criterion


def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps == "":
            return None
        scheduler = lr_scheduler.MultiStepLR(optimizer,  # 这里记得看一下
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'none':
        scheduler = None
    return scheduler


def main(config):
    # SummaryWriter画图用的
    with SummaryWriter(comment='_{}_{}'.format(config.arch, config.dataset)) as writer:
        # 选择datasets中的cifar10
        dataset_config = datasets.FPN(config) if config.dataset == 'FPN' else datasets.cifar10()
        # dataset_config = datasets.cifar10() if config.dataset == 'cifar10' else datasets.cifar100()
        # num_classes为类别数
        # num_classes = dataset_config.pop('num_classes')
        train_loader, eval_loader = data_loaders(**dataset_config, config=config)

        dummy_input = torch.randn(1, 1, 200, 200)  # 添加一个模型的图
        # fpnmodel = arch_n[config.arch]
        net = Fpn_n()
        writer.add_graph(net, dummy_input)

        device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = create_loss_fn(config)
        if config.is_parallel:
            net = torch.nn.DataParallel(net).to(device1)
        else:
            device1 = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu'
            net = net.to(device1)
        optimizer = create_optim(net.parameters(), config)
        trainer = Trainer.PseudoLabel(net, optimizer, criterion, device1, config, writer, save_dir='./model')
        scheduler = create_lr_scheduler(optimizer, config)
        trainer.loop(config.epochs, train_loader, eval_loader,
                     scheduler=scheduler, print_freq=config.print_freq)
