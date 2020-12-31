#!coding:utf-8
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from util.datasets import NO_LABEL


class PseudoLabel:

    def __init__(self, model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.list_path2 = config.list_path2
        self.list_path1 = config.list_path1
        self.global_step = 0  # 全局训练数据的次数
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2
        self.af = config.af
        self.upsize = config.upsize

    def outputs(self, data):
        outputs_feature = self.model(data.float())
        outputs = F.interpolate(outputs_feature, size=(int(self.upsize), int(self.upsize)), mode='bilinear',
                                align_corners=True)
        outputs = outputs * 100 + 298
        return outputs

    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        # accuracy = []
        # labeled_n = 0
        mode = "train" if is_train else "test"
        if type(data_loader) == dict:
            data_loader = zip(data_loader['l'], data_loader['ul'])
        for batch_idx, (data, targets) in enumerate(data_loader):  # 将可遍历的对象组成索引序列，并标出数据和下标
            self.global_step += batch_idx
            if is_train:
                if self.list_path2 != 'None' and self.list_path1 != 'None':  # 半监督
                    data_label = data[0]
                    targets_label = data[1]
                    data_unlabel = targets[0]
                    targets_unlabel = targets[1]
                    data_label, data_unlabel, targets_label, targets_unlabel = data_label.to(
                        self.device), data_unlabel.to(
                        self.device), targets_label.to(self.device), targets_unlabel.to(self.device)
                    outputs_l = self.outputs(data_label)
                    outputs_ul = self.outputs(data_unlabel)
                    # labeled_bs = self.labeled_bs
                    labeled_loss = self.loss_fn(outputs_l, targets_label.float())
                    # labeled_loss = torch.sum(self.loss_fn(outputs_l, targets_label.float())) / labeled_bs
                    with torch.no_grad():
                        outputs_ulp = self.outputs(data_unlabel)
                        pseudo_labeled = outputs_ulp
                    unlabeled_loss = self.loss_fn(outputs_ul, pseudo_labeled.float())
                    loss = labeled_loss + self.unlabeled_weight() * unlabeled_loss
                elif self.list_path2 != 'None' and self.list_path1 == 'None':  # 有监督
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.outputs(data)
                    labeled_loss = self.loss_fn(outputs, targets.float())
                    unlabeled_loss = torch.Tensor([0])
                    loss = labeled_loss
                elif self.list_path2 == 'None' and self.list_path1 != 'None':  # 无监督
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.outputs(data)
                    unlabeled_loss = self.loss_fn(outputs, targets.float())
                    labeled_loss = torch.Tensor([0])
                    loss = unlabeled_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                data, targets = data.to(self.device), targets.to(self.device)  # 放到GPU上
                outputs = self.outputs(data)
                # labeled_bs = data.size(0)
                labeled_loss = unlabeled_loss = torch.Tensor([0])
                loss = self.loss_fn(outputs, targets.float())
            # labeled_n += labeled_bs
            loop_loss.append(loss.item())
            # acc = loss.item()
            # targets.eq(outputs.max(1)[1]).sum().item()
            # accuracy.append(acc)
            if print_freq > 0 and (batch_idx % print_freq) == 0:
                print(
                    f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.3f}\t unlabeled loss: {unlabeled_loss.item():.3f}\t loss: {loss.item():.3f}")
            if self.writer:
                self.writer.add_scalar(mode + '_global_loss', loss.item(), self.global_step)
                # self.writer.add_scalar(mode + '_global_accuracy', acc / labeled_bs, self.global_step)
        loop_loss = np.array(loop_loss).mean()
        print(f">>>[{mode}]loss\t loss: {loop_loss:.3f}")
        if self.writer:
            self.writer.add_scalar(mode + '_epoch_loss', loop_loss, self.epoch)

        return loop_loss

    # def split_list(self, data, targets, iflabel):
    #     data_label = []
    #     data_unlabel = []
    #     targets_label = []
    #     targets_unlabel = []
    #     for index in range(len(iflabel)):
    #         if iflabel[index] == 1:
    #             data_label.append(data[index:index + 1, :, :, :])
    #             targets_label.append(targets[index:index + 1, :, :, :])
    #         elif iflabel[index] == 0:
    #             data_unlabel.append(data[index:index + 1, :, :, :])
    #             targets_unlabel.append(targets[index:index + 1, :, :, :])
    #     data_label = torch.cat(data_label, dim=0)
    #     targets_label = torch.cat(targets_label, dim=0)
    #     data_unlabel = torch.cat(data_unlabel, dim=0)
    #     targets_unlabel = torch.cat(targets_unlabel, dim=0)
    #     return data_label, data_unlabel, targets_label, targets_unlabel

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch - self.T1) / (self.T2 - self.T1) * self.af
            if self.epoch > self.T2:
                alpha = self.af
        return alpha

    def train(self, data_loader, print_freq=20):
        self.model.train()  # 启用dropout和batchnormalization
        with torch.enable_grad():
            loss = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()  # 不启用dropout和batchnormalization
        with torch.no_grad():
            loss = self._iteration(data_loader, print_freq, is_train=False)
        return loss

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):  # 开始循环训练，epochs 训练数据、测试数据
        minloss = 10000000
        for ep in range(epochs):
            self.epoch = ep  # ep表示当次迭代数
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)  # print_freq为输出的频率？
            print("------ Testing epochs: {} ------".format(ep))
            loss = self.test(test_data, print_freq)
            if scheduler is not None:
                scheduler.step()
            # if ep % self.save_freq == 0:
            if loss < minloss:
                minloss = loss
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                     "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            torch.save(state, model_out_path / f"model.pth")
