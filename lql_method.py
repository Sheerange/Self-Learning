from tqdm import tqdm
import sys
from torch.optim.optimizer import Optimizer, required
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as FF
import math
import torch
from torch.optim import Adam
import random
import pandas as pd
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
"val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}
transform = data_transform['train']

def consistency_loss(output1, output2):
    return torch.mean((output1 - output2) ** 2)

def Cbalanced_CE_loss(logits, targets):
    # logits: 模型的输出，形状为 (N, C)
    # targets: 真实标签，形状为 (N, )

    # 计算 softmax
    probs = F.softmax(logits, dim=1)

    # 将 targets 转换为 one-hot 编码
    targets_one_hot = torch.zeros_like(logits)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

    # 计算交叉熵损失
    loss = - torch.sum(torch.sum(targets_one_hot * torch.log(probs), dim=0) / torch.clamp(torch.sum(targets_one_hot, dim=0), min=1)) / logits.shape[1]
    return loss


class Tco(nn.Module):
    def __init__(self, net, validate_loader, max_confidence=0.99, min_confidence=0.9, margin=0.3, lr=1e-4, tag='max', confidence=0.9):
        super(Tco, self).__init__()
        print("Already use Tco method")
        self.max_confidence = max_confidence
        self.min_confidence = min_confidence
        self.confidence = confidence
        self.margin = margin
        self.criterion = nn.CrossEntropyLoss()
        # self.solver_other = torch.optim.SGD(net.parameters(), lr=lr)
        self.solver_other = torch.optim.Adam(net.parameters(), lr=lr)
        self.validate_loader = validate_loader
        self.softmax_ = nn.Softmax(dim=1)
        self.tag = tag
        # self.dic_select = {i: None for i in range(100)}
        self.dic_select = {'label_pseudo': [],
                           'img': [],
                           'confidence': [],
                           'label': [],
                           }
        self.current_step = 0

    def selection(self, imgs, labels, net, dict):
        net.eval()                     #  汇总
        with torch.no_grad():
            for k, i in enumerate(imgs):
                i = i.unsqueeze(0)
                output = net(i)
                output = self.softmax_(output)
                _, pred = torch.max(output, 1)
                dict['label_pseudo'].append(pred.to('cpu'))
                dict['img'].append(i.to('cpu'))
                dict['confidence'].append(output.max().to('cpu'))
                dict['label'].append(labels[k].to('cpu'))
        return dict

    def select(self, imgs, net, labels, allocated):
        net.eval()
        ret = []
        pseudo_labels = []
        corrects = {key: 0 for key in range(len(net.state_dict()['fc.weight']))}
        er = 0
        sum_label = 0
        # weight = net.state_dict()['fc.weight']
        # value = torch.norm(weight, dim=1)
        # min_value = torch.norm(weight,dim=1).min()
        # max_value = torch.norm(weight,dim=1).max()
        # confidence_classes = self.min_confidence + ((value - min_value) / (max_value - min_value)) * (self.max_confidence - self.min_confidence) # 自适应类别置信度

        allocated = [i / max(allocated) for i in allocated]
        with torch.no_grad():
            for k, i in enumerate(imgs):
                i = i.reshape([1, i.shape[0], i.shape[1], i.shape[2]])
                output = net(i)
                output = (output.squeeze(0) * torch.tensor(allocated).to('cuda')).unsqueeze(0)
                output = self.softmax_(output)
                _, pred = torch.max(output, 1)
                pred = pred.unsqueeze(0)
                # if _[0] > self.s_l:    # 基于置信度
                if self.tag == 'margin':
                    condition = torch.sort(output, descending=True).values[:, 0] - torch.sort(output, descending=True).values[:,1] > self.margin
                elif self.tag == 'max':
                    # condition = _[0] > confidence_classes[pred.squeeze(0)[0]]
                    condition = _[0] > (self.max_confidence + self.min_confidence) / 2
                if condition:  # 基于置信度的margin
                    er += 1
                    if er == 1:
                        ret = i
                        pseudo_labels = pred
                    elif er >= 2:
                        ret = torch.cat((ret, i), 0)
                        pseudo_labels = torch.cat((pseudo_labels, pred), 0)
                    if pred == labels[k]:
                        sum_label += 1
                        corrects[pred.item()] += 1
        if er >= 2:
            er = True
        else:
            er = False
        # print("pseudo labels:{}".format(pseudo_labels))
        # print("labels:{}".format(labels))
        return ret, er, pseudo_labels, sum_label, corrects

    def forward(self, net, new_lr, alpha, allocated, batch_size, optimizer, scheduler, current_step, total_steps, topN=10, mix=False):
        # for param_group in self.solver_other.param_groups:
        #     param_group['lr'] = new_lr * alpha

        val_bar = tqdm(self.validate_loader, file=sys.stdout)
        length = 0
        sums_label = 0
        class_pseudo = {key: 0 for key in range(len(net.state_dict()['fc.weight']))}
        class_correct = {key: 0 for key in range(len(net.state_dict()['fc.weight']))}
        allocated = allocated
        self.dic_select = {'label_pseudo': [],
                           'img': [],
                           'confidence': [],
                           'label': [],
                           }
        for imgs, labels in val_bar:
            # self.solver_other.zero_grad()
            imgs = imgs.to('cuda')
            labels = labels.to('cuda')
            # imgs,er,pseudo_labels,sum_label,corrects = self.select(imgs,net,labels,allocated)
            self.dic_select = self.selection(imgs, labels, net, self.dic_select)

            # consistency
            # net.train()
            # optimizer.zero_grad()
            # output = net(imgs)
            # noise = torch.randn_like(imgs) * 0.1
            # imgs_noise = imgs + noise
            # output_noisy = net(imgs_noise)
            # consistency_reg_loss = consistency_loss(output_noisy, output)
            # consistency_reg_loss.backward()
            # optimizer.step()
            #
            # val_bar.desc = "consistency_loss:{:.3f}".format(consistency_reg_loss)

        df = pd.DataFrame(self.dic_select)

        # df['label_pseudo'] = df['label_pseudo'].astype(int)
        # df['label'] = df['label'].astype(int)
        # df['confidence'] = df['confidence'].astype(float)
        # print('len(df): {}'.format(len(df)))
        # df.to_csv('df.csv')

        imgs = []
        labels_pseudo = []          # 伪标签列表
        labels= []

        # 基于类别topN分配伪标签
        # corrects_class = {i: 0 for i in range(len(net.state_dict()['fc.weight']))}
        # for i in range(len(net.state_dict()['fc.weight'])):  # 选择每个类别topN置信度的图片样本
        #     # for j in df[df['label'] == i]['confidence'].astype(float).nlargest(topN).index:      #根据标签选择类别
        #     for j in df[df['label_pseudo'] == i]['confidence'].astype(float).nlargest(topN).index:    #根据伪标签选择类别
        #         imgs.append(df.iloc[j]['img'].squeeze(0))
        #         # labels_pseudo.append(df.iloc[j]['label_pseudo'].squeeze(0).cpu().numpy())       # 伪标签列表扩充
        #         # labels.append(df.iloc[j]['label'].squeeze(0).cpu().numpy())  # 标签列表扩充
        #         labels_pseudo.append(df.iloc[j]['label_pseudo'].squeeze(0))  # 伪标签列表扩充
        #         labels.append(df.iloc[j]['label'].squeeze(0))  # 标签列表扩充
        #
        #     corrects_class[i] = np.sum(np.array(labels_pseudo[topN*i : topN*(i+1)]) == np.array(labels[topN*i : topN*(i+1)]))
        #     print('class {} right number of pseudo label:{}'.format(i, corrects_class[i]))      # 打印伪标签的正确个数

        # 基于统一置信度分配伪标签
        for j in df[df['confidence']>self.confidence].index:    # 根据统一置信度来选择
            imgs.append(df.iloc[j]['img'].squeeze(0))
            labels_pseudo.append(df.iloc[j]['label_pseudo'].squeeze(0).cpu().numpy())       # 伪标签列表扩充
            labels.append(df.iloc[j]['label'].squeeze(0).cpu().numpy())  # 标签列表扩充

        corrects = np.sum(np.array(labels_pseudo) == np.array(labels))
        print('The right number of pseudo label:{}, total number:{}'.format(corrects,len(labels_pseudo)))      # 打印伪标签的正确个数
        for i in range(len(net.state_dict()['fc.weight'])):
            correct = np.sum((np.array(labels_pseudo) == np.array(labels)) & (np.array(labels_pseudo)==i))
            total = sum(np.array(labels_pseudo)==i)
            print('class {}: correct {}  total {}'.format(i, correct,total))

        if len(imgs) == 0:
            return net
        imgs = torch.stack(imgs).squeeze(1)

        labels = [i.item() for i in labels]
        labels = torch.tensor(labels)
        labels_pseudo = [i.item() for i in labels_pseudo]
        labels_pseudo = torch.tensor(labels_pseudo)

        class_weights = [0]*len(net.state_dict()['fc.weight'])
        for i in range(len(net.state_dict()['fc.weight'])):
            class_weights[i] = 1 / max(1,float(sum(np.array(labels_pseudo)==i)))
        weights = torch.tensor(class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=weights.to('cuda'))
        print(weights)
        if not mix:      # 是否混合训练？
            net.train()
            sum_loss=0
            train_bar = tqdm(range(0, imgs.shape[0], batch_size), file=sys.stdout)
            # for i in tqdm(range(0, imgs.shape[0], batch_size), file=sys.stdout):
            for i in train_bar:
                batch = imgs[i : i+ batch_size].to('cuda')
                label = labels[i: i + batch_size].to('cuda')
                label_pseudo = labels_pseudo[i: i + batch_size].to('cuda')
                output = net(batch)
                _, pred = torch.max(output, 1)
                # self.solver_other.zero_grad()
                optimizer.zero_grad()
                # loss = self.criterion(output, label_pseudo)       # label or pred?    label 就是有监督训练，pred 就是半监督训练  应该用pred还是label_pseudo?   pred是在变化的， 而label_pseudo是每一轮都确定的
                loss = Cbalanced_CE_loss(output, label_pseudo)

                loss.backward()
                # self.solver_other.step()
                optimizer.step()
                if self.current_step < total_steps:
                    scheduler.step()         # OneCycleLR
                    self.current_step += 1
                sum_loss += loss.item()
                train_bar.desc = "tco loss:{:.3f}".format(loss)
            print("TCO loss:{:.3f}".format(sum_loss/len(train_bar)))
            return net
        return imgs, labels_pseudo