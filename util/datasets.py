import os
import itertools
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import VisionDataset
import scipy.io as sio

NO_LABEL = 0
LABEL = 1


def FPN(config):
    transform_layout = transforms.Compose(
        [
            # transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([config.mean_layout]),
                torch.tensor([config.std_layout]),
            ),
        ]
    )
    heat_transform = transforms.Compose(
        [
            # transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
        ]
    )
    if config.list_path1 != 'None' and config.list_path2 != 'None':
        with open(config.list_path1) as rf:
            count1 = 0  # 无标签数
            for index, line in enumerate(rf):
                count1 += 1
        with open(config.list_path2) as rf:
            count2 = 0  # 有标签数
            for index, line in enumerate(rf):
                count2 += 1
        batch_size_ul = int(count1 / count2 * config.batch_size)
    else:
        batch_size_ul = config.batch_size

    return {
        'train_transform': transform_layout,
        'eval_transform': transform_layout,
        'heat_transform': heat_transform,
        'datadir': '/mnt/layout_data/v0.3/data',
        'batch_size': config.batch_size,
        'batch_size_ul': batch_size_ul
    }


# 图像处理，翻转加标准化
# def cifar10():
#     channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
#                          std=[0.2470, 0.2435, 0.2616])
#     # train_transform为训练前图像翻转
#     train_transform = transforms.Compose([  # compose组合多个transform操作
#         RandomTranslateWithReflect(4),  # 4为最大翻转数
#         transforms.RandomHorizontalFlip(),  # 水平翻转
#         transforms.ToTensor(),  # 把图片变为tensor
#         transforms.Normalize(**channel_stats)  # 标准化Tensor
#     ])
#     eval_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(**channel_stats)
#     ])
#
#     return {
#         'train_transform': train_transform,
#         'eval_transform': eval_transform,
#         'datadir': './data-local/images/cifar/cifar10/by-image',
#         'num_classes': 10
#     }


# def relabel_dataset(dataset, labels):
#     unlabeled_idxs = []  # 列表
#     for idx in range(len(dataset.sample_files)):  # 数据中的长度，排序，即有多少个数据
#         path = dataset.sample_files[idx]  # 对其中每个数据得到路径
#         filename = os.path.basename(path)  # 返回路径中文件名
#         if filename in labels:  # 若该文件名在有标记的数据文件列表中
#             dataset.sample_files[idx] = path  # 同时将路径和代号重新赋给文件路径中
#             labels.remove(filename)
#         else:
#             dataset.sample_files[idx] = path
#             unlabeled_idxs.append(idx)
#
#     if len(labels) != 0:  # 如果有标签列表长度不为0
#         message = "List of unlabeled contains {} unknow files: {}, ..."  #
#         some_missing = ', '.join(labels[:5])  #
#         raise LookupError(message.format(len(labels), some_missing))
#
#     labeled_idxs = sorted(set(range(len(dataset.sample_files))) - set(unlabeled_idxs))
#     return labeled_idxs, unlabeled_idxs


# class TwoStreamBatchSampler(Sampler):  # 同时迭代两个列表
#     """Iterate two sets of indices
#     """
#
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.primary_batch_size = batch_size - secondary_batch_size
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#
#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0
#
#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             secondary_batch + primary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                    grouper(secondary_iter, self.secondary_batch_size))
#         )
#
#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size
#
#
# def iterate_once(iterable):
#     return np.random.permutation(iterable)  # 随机排列一个序列或数组
#
#
# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#
#     return itertools.chain.from_iterable(infinite_shuffles())  # 将多个迭代器高效连接
#
#
# def grouper(iterable, n):
#     args = [iter(iterable)] * n
#     return zip(*args)


class dataloader(VisionDataset):
    def __init__(self,
                 root,
                 list_path,
                 load_name='f',
                 resp_name='u',
                 extensions='mat',
                 transform=None,
                 target_transform=None,
                 is_valid_file=None,
                 max_iters=None, ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.loader = mat_loader
        self.list_path = list_path
        self.load_name = load_name
        self.resp_name = resp_name
        self.extensions = extensions
        # self.sample_files = make_dataset(root, extensions, is_valid_file)
        self.sample_files = make_dataset_list(root, list_path, extensions, is_valid_file,
                                              max_iters=max_iters)

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name)
        if self.transform is not None:
            load = load.astype(float)
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return len(self.sample_files)


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None, max_iters=None):
    """make_dataset() from torchvision.
        """
    files = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf:  # 打开list_path并读取其中内容
        for line in rf.readlines():
            data_path = line.strip()  # 移除字符串头尾的制定字符，此处为空格
            path = os.path.join(root_dir, data_path)  # path表示每个数据的路径
            if is_valid_file(path):  # 在files中加入读取的数据路径
                files.append(path)
    if max_iters is not None:  # max_iters非空，则取到最大数据处
        files = files * int(np.ceil(float(max_iters) / len(files)))
        files = files[:max_iters]
    return files  # files中存放数据的路径


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)  # 以extensions为后缀的文件小写


def mat_loader(path, load_name, resp_name=None):
    mats = sio.loadmat(path)
    load = mats.get(load_name)
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp  # load存放F即布局，resp存放u即温度场标签
