import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import numpy as np

# def compute_attention_head(x, proxies, fc_shared):
#     q = fc_shared(x)  # batchsize, output_fc_shared
#     k = fc_shared(proxies)  # num_class, output_fc_shared
#     scale = q.shape[1] ** -0.5
#     q_k = F.softmax(torch.matmul(q, k.transpose(0, 1))*scale, dim=1)  # batchsize, num_class
#     q_k_v = torch.matmul(q_k, k)  # batchsize, output_fc_shared
#     return q_k_v
def z_score_normalization(tensor):
    mean = torch.mean(tensor)
    std_dev = torch.std(tensor)
    z_scores = (tensor - mean) / std_dev
    return z_scores

def compute_attention_head(x, proxies, fc_q, fc_k, fc_v, bn_q, bn_k, bn_v):           #BatchNormalization 用在q,k,v之后
# def compute_attention_head(x, proxies, fc_q, fc_k, fc_v, bn_p, bn_x):                   #BatchNormalization 在q,k,v之前
    # proxies = bn_p(proxies)
    # x = bn_x(x)
    q = fc_q(x)  # batchsize, output_fc_shared
    q = bn_q(q)
    k = fc_k(proxies)  # num_class, output_fc_shared
    k = bn_k(k)
    v = fc_v(proxies)
    v = bn_v(v)

    scale = q.shape[1] ** -0.5
    q_k = F.softmax(torch.matmul(q, k.transpose(0, 1)) * scale, dim=1)  # batchsize, num_class
    # q_k = F.softmax(torch.matmul(q, k.transpose(0, 1)), dim=1)  # softmax不带scale
    # q_k = torch.matmul(q, k.transpose(0, 1))          # 无softmax
    # q_k = F.softmax(z_score_normalization(torch.matmul(q, k.transpose(0, 1))), dim=1)  # softmax + z-score
    # q_k = z_score_normalization(torch.matmul(q, k.transpose(0, 1)))  # z-score

    q_k_v = torch.matmul(q_k, v)  # batchsize, output_fc_shared
    torch.save(q,'function_q.pt')
    torch.save(k, 'function_k.pt')
    torch.save(v, 'function_v.pt')
    torch.save(q_k, 'function_qk.pt')
    return q_k_v

def forward_aug(net, x, proxies, pred=None):
    # proxies = net.fc.weight.data
    # proxies = net._fc.weight.data    # efficientNet
    # proxies = net.classifier.weight.data  # DenseNet
    pseudo_prototypes = torch.stack([proxies[i] for i in pred])

    attention_outputs = [
        compute_attention_head(x=x, proxies=proxies, fc_q=fc_q, fc_k=fc_k, fc_v=fc_v, bn_q=bn_q, bn_k=bn_k, bn_v=bn_v)
        for fc_q, fc_k, fc_v, bn_q, bn_k, bn_v in zip(net.fc_q, net.fc_k, net.fc_v, net.bn_q, net.bn_k, net.bn_v)]
    f_cat = torch.cat(attention_outputs, dim=1)  # Concatenate along the feature dimension
    f_a = F.relu(net.fc_a(f_cat))  # batchsize, output_fc_a
    x = F.relu(net.fc_r(f_a) + pseudo_prototypes)  # batchsize, channel

    return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.n=512
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            #self.fc=nn.Linear(self.n,5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x, pred=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.aug or pred is None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)     # x = batchsize, 2048
            x = self.fc(x)          #  x = batchsize, 2048
        #
        if self.aug and pred is not None:
            self.proxies = self.fc.weight.data
            pseudo_prototypes = torch.stack([self.proxies[i] for i in pred])
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # batchsize, channel

            attention_outputs = [compute_attention_head(x=x, proxies=self.proxies, fc_q=fc_q, fc_k=fc_k, fc_v=fc_v, bn_q=bn_q, bn_k=bn_k, bn_v=bn_v)
                                 for fc_q, fc_k, fc_v, bn_q, bn_k, bn_v in zip(self.fc_q, self.fc_k, self.fc_v, self.bn_q, self.bn_k, self.bn_v)]
            f_cat = torch.cat(attention_outputs, dim=1)  # Concatenate along the feature dimension
            f_a = F.relu(self.fc_a(f_cat))  # batchsize, output_fc_a

            f_r = F.relu(self.fc_r(f_a) + pseudo_prototypes)  # batchsize, channel
            # f_r = F.relu(self.fc_r(f_a) + x)  # batchsize, channel
            # f_r = F.relu(self.fc_r(f_a))  # batchsize, channel      no residual connection

            # torch.save(x, 'x.pt')
            # torch.save(f_cat, 'f_cat.pt')
            # torch.save(f_a, 'f_a.pt')
            # torch.save(self.fc_r(f_a),'f_r.pt')
            # torch.save(pseudo_prototypes, 'pseudo_prototypes.pt')
            x = self.fc(f_r)

        # if self.aug and pred is not None:
        #     self.proxies = self.fc.weight.data
        #     pseudo_prototypes = torch.stack([self.proxies[i] for i in pred])
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)  # batchsize, channel
        #     x = x*1/2 + pseudo_prototypes*1/2
        #     x = self.fc(x)

        return x,torch.zeros_like(x)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    # "https://download.pytorch.org/models/resnet152-f82ba261.pth"
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
