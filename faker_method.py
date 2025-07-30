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
from torch.utils.data import Dataset, DataLoader

def generate_pseudo_labels(model, dataloader, device, threshold=0.95, threshold_negative=0.01, lambda_aug=0.5):
    model.eval()
    pseudo_labels = []
    high_confidence_data = []
    trully_labels = []

    corrects = 0
    total_pseudo_labels = 0


    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.aug = False
            output1, _ = model(inputs)
            probs = torch.softmax(output1, dim=1)

            # 特征增强
            _, pred = torch.max(output1, 1)
            model.aug = False # 是否特征增强
            output2, _ = model(inputs, pred)
            probs = torch.softmax(output2, dim=1)


            confidence, predicted = torch.max(probs, dim=1)
            # print(confidence)
            high_confidence_mask = confidence > threshold
            high_confidence_data.append(inputs[high_confidence_mask])
            pseudo_labels.append(predicted[high_confidence_mask])
            trully_labels.append(labels[high_confidence_mask])
            # print(pseudo_labels)
            corrects += (predicted[high_confidence_mask] == labels[high_confidence_mask]).sum().item()
            total_pseudo_labels += high_confidence_mask.sum().item()


        print("right pseudo labels:{}    total pseudo labels:{}".format(corrects, total_pseudo_labels))

    if high_confidence_data:
        high_confidence_data = torch.cat(high_confidence_data)
        pseudo_labels = torch.cat(pseudo_labels)
        trully_labels = torch.cat(trully_labels)
    else:
        high_confidence_data = torch.empty((0, 1, 28, 28), device=device)
        pseudo_labels = torch.empty((0,), dtype=torch.long, device=device)
        trully_labels = torch.empty((0,), dtype=torch.long, device=device)

    return high_confidence_data, pseudo_labels, trully_labels
    # return high_confidence_data, pseudo_labels, low_confidence_data, negative_labels

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CombinedMixupDataset(Dataset):
    def __init__(self, dataset1, dataset2, alpha=1.0): # alpha 取大于等于1，才会出现混合样本
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.alpha = alpha

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        x1, y1 = self.dataset1[idx]
        idx2 = np.random.randint(0, len(self.dataset2))
        x2, y2 = self.dataset2[idx2]

        # 确保混合的是同标签
        while y1 != y2:
            idx2 = np.random.randint(0, len(self.dataset2))
            x2, y2 = self.dataset2[idx2]

        # lam = np.random.beta(self.alpha, self.alpha)
        lam = 0.5
        x = lam * x1 + (1 - lam) * x2
        y = y1  # 保持标签不变

        return x, y

def Negative_CE_loss(logits, labels):
    probs = F.softmax(logits, dim=1)
    # targets_multi_hot = torch.zeros_like(probs)
    # targets_multi_hot.scatter_(1, labels, 1.0)
    # loss = - torch.sum((1-targets_multi_hot) * torch.log(1-probs)) / torch.sum(targets_multi_hot, dim=1)
    loss = - torch.sum( torch.sum((1 - labels) * torch.log(1 - probs), dim=1) / torch.sum(labels, dim=1)) / logits.shape[0]
    return loss

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
    def __init__(self, net, pseudo_loader, train_dataset, confidence=0.9, lambda_aug=0.5):
        super(Tco, self).__init__()
        print("Already use Tco method")
        self.confidence = confidence
        self.pseudo_loader = pseudo_loader
        self.softmax_ = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataset = train_dataset
        self.lambda_aug = lambda_aug

    def forward(self, net, batch_size, optimizer, nw=20):

        # 生成伪标签
        high_confidence_data, pseudo_labels, trully_labels = generate_pseudo_labels(net, self.pseudo_loader, 'cuda', threshold=self.confidence, lambda_aug=self.lambda_aug)
        pseudo_output = pseudo_labels  # 返回的伪标签
        trully_output = trully_labels  # 返回的真实标签

        if len(high_confidence_data) == 0:
            return net, pseudo_output, trully_output

        pseudo_dataset = torch.utils.data.TensorDataset(high_confidence_data.cpu(), pseudo_labels.cpu(), trully_labels.cpu())
        pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        # 创建 CombinedMixupDataset 数据集
        combined_mixup_dataset = CombinedMixupDataset(pseudo_dataset, self.train_dataset, alpha=1.5)
        combined_loader = torch.utils.data.DataLoader(combined_mixup_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

        # pseudo label training
        pseudo_bar = tqdm(pseudo_loader, file=sys.stdout)
        combined_bar = tqdm(combined_loader, file=sys.stdout)
        sum_loss = 0
        net.train()
        for data in pseudo_bar:
        # for data in combined_bar:
            imgs, pseudo_labels, trully_labels = data
            if imgs.shape[0] == 1:
                break;
            imgs = imgs.to('cuda')
            pseudo_labels = pseudo_labels.to('cuda')
            trully_labels = trully_labels.to('cuda')
            optimizer.zero_grad()

            net.aug = False
            output, _ = net(imgs)

            net.aug = True
            _, pred = torch.max(output, 1)
            # output_aug, _ = net(imgs, pred)
            output_aug, _ = net(imgs, pseudo_labels)        # 为什么不是pseudo_labels???
            log_probs = F.log_softmax(output, dim=1)
            probs_aug = F.softmax(output_aug, dim=1)

            loss = Cbalanced_CE_loss(output, pseudo_labels) + (self.lambda_aug) * (Cbalanced_CE_loss(output_aug, pseudo_labels) + F.kl_div(log_probs,probs_aug,reduction='batchmean'))  # 弱图像增强预测概率与伪标签的cross_entropy + 特征增强预测概率与伪标签的cross_entropy
            # loss = self.criterion(output, pseudo_labels) + (self.lambda_aug) * (
            #         self.criterion(output_aug, pseudo_labels) + F.kl_div(log_probs, probs_aug,
            #                                                                 reduction='batchmean'))  # CE loss
            # loss = self.criterion(output, pseudo_labels) + (self.lambda_aug) * self.criterion(output_aug, pseudo_labels)  # 弱图像增强预测概率与伪标签的cross_entropy + 特征增强预测概率与伪标签的cross_entropy
            # loss = self.criterion(output, pseudo_labels)  # 弱图像增强预测概率与伪标签的cross_entropy + 特征增强预测概率与伪标签的cross_entropy
            # loss = Cbalanced_CE_loss(output, pseudo_labels)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            pseudo_bar.desc = "tco loss:{:.3f}".format(loss)
        print("TCO loss:{:.7f}".format(sum_loss / len(pseudo_bar)))
            # combined_bar.desc = "tco loss:{:.3f}".format(loss)

        return net, pseudo_output, trully_output