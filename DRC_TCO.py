import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from torchvision import transforms, datasets
from model import resnet34
from model import resnet50
from model import resnet101
from model import resnet152
from model import resnext50_32x4d
from resnest import resnest50
from efficientnet_pytorch import EfficientNet
from model_dense import densenet121, load_state_dict
from pvt import pvt_small
from cls_cvt import ConvolutionalVisionTransformer
from swin_transformer import SwinTransformer
from utils import load_pretrained
from poolformer import poolformer_s24

from pyramid_vig import pvig_ti_224_gelu
from gcn_lib import Grapher, act_layer

from transnext import transnext_tiny

import time
from torch.optim.lr_scheduler import OneCycleLR
# from lql_method import Tco
from faker_method import Tco
import copy
from torch.nn import functional as F
import torch.nn.init as init

import random
import PIL
from PIL import Image, ImageEnhance, ImageOps


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--tco", type=int, default=1)
parser.add_argument("--threshold_acc", type=int, default=0)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dataset", type=str, default='cifar10_tiny')
parser.add_argument("--confidence", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--fc_qkv", type=int, default=256)
parser.add_argument("--fc_a", type=int, default=256)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--lambda_aug", type=float, default=0.5)
parser.add_argument("--epoch_start", type=int, default=4)
parser.add_argument("--rewind_threshold", type=int, default=5)
args = parser.parse_args()

print('acc:{}'.format(args.threshold_acc))
print('confidence:{}'.format(args.confidence))
print('lr:{}'.format(args.lr))

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        init.zeros_(m.bias)

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)
def main():
    # CIFAR-10: mean: [0.49139968, 0.48215841, 0.44653091], std: [0.24703223, 0.24348513, 0.26158784]
    # CIFAR-100: mean: [0.50707516, 0.48654887, 0.44091784], std: [0.26733429, 0.25643846, 0.27615047]
    # ImageNet: mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                   ]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])]),
    }

    def _loss_(net, data_loader):
        net.eval()
        loss_total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')

                net.aug = False
                output, _ = net(imgs)

                loss = criterion(output, labels.to('cuda'))
                loss_total += loss

        return loss_total / len(data_loader)


    def _accuracy_(net, data_loader):
        net.eval()
        num_total = 0
        num_acc = 0
        num_acc_aug = 0
        num_acc_top5 = 0
        num_acc_aug_top5 = 0
        with torch.no_grad():
            for imgs, labels in data_loader:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')

                # Top-1 accuracy without augmentation
                net.aug = False
                output, _ = net(imgs)
                _, pred = torch.max(output, 1)
                net.aug = True

                # Top-1 accuracy with augmentation
                output_aug, _ = net(imgs, pred=pred)
                _, pred_aug = torch.max(output_aug, 1)

                # Top-1 accuracy
                num_acc += torch.sum(pred == labels.detach_())
                # Top-1 accuracy after augmentation
                num_acc_aug += torch.sum(pred_aug == labels.detach_())

                # Calculate Top-5 accuracy
                _, top5_pred = torch.topk(output, 5, dim=1)
                top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))
                num_acc_top5 += torch.sum(top5_correct)

                num_total += labels.size(0)

            LV = num_acc.detach().cpu().numpy() * 100 / num_total
            LV_aug = num_acc_aug.detach().cpu().numpy() * 100 / num_total
            LV_top5 = num_acc_top5.detach().cpu().numpy() * 100 / num_total

        return LV, LV_aug, LV_top5

    epochs = args.epochs
    base_lr = args.lr
    momentum = 0
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    nw = args.nw

    model_name = 'resNet50'
    # model_name = 'transnext-tiny'
    # model_name = 'resNet152'
    # model_name = 'resNeSt50'
    # model_name = 'resNeXt50'
    # model_name = 'efficientnet-b3'
    # model_name = 'densenet121'
    # model_name = 'PVT-s'
    # model_name = 'CVT-13'
    # model_name = 'swin-ti'
    # model_name = 'poolformer-s24'
    # model_name = 'pvig-ti'
    # model_name = 'transNext-ti'

    dataset_name = args.dataset
    # save_path = 'weights/' + model_name + '.pth'

    model_weight_path = './resnet50-pre.pth'
    # model_weight_path = './resnet152-pre.pth'
    # model_weight_path = './resnext50-pre.pth'
    # model_weight_path = './resnest50-pre.pth'
    # model_weight_path = './efficientnet-b3.pth'
    # model_weight_path = './densenet121.pth'
    # model_weight_path = './pvt_small.pth'
    # model_weight_path = './CvT-13.pth'
    # model_weight_path = './swin_tiny_patch4_window7_224.pth'
    # model_weight_path = "./poolformer_s24.pth.tar"
    # model_weight_path = './pvig_ti_78.5.pth.tar'
    # model_weight_path = './transnext_tiny_224_1k.pth'


    # channel = 1536           # efficientNet-b3: 1536   # denseNet-121: 1024; PVT-s:512; Swin:768; CVT:384; poolformer:512; pvig:1024; transNext:576
    if model_weight_path == './resnet152-pre.pth' or model_weight_path == './resnext50-pre.pth' or model_weight_path == './resnest50-pre.pth' or model_weight_path == './resnet50-pre.pth':
        channel = 2048
    elif model_weight_path == './efficientnet-b3.pth':
        channel = 1536
    elif model_weight_path == './densenet121.pth':
        channel = 1024
    elif model_weight_path == './pvt_small.pth':
        channel = 512
    elif model_weight_path == './CvT-13.pth':
        channel = 384
    elif model_weight_path == './swin_tiny_patch4_window7_224.pth':
        channel = 768
    elif model_weight_path == "./poolformer_s24.pth.tar":
        channel = 512
    elif model_weight_path == './pvig_ti_78.5.pth.tar':
        channel = 1024
    elif model_weight_path == './transnext_tiny_224_1k.pth':
        channel = 576


    num_classes = 100 if dataset_name == ('cifar100' or 'cifar311') else 200

    # The confidence threshold
    threshold_acc = args.threshold_acc
    dataset = args.dataset
    confidence = args.confidence

    # the feature augmentation
    output_fc_qkv = args.fc_qkv
    output_fc_a = args.fc_a

    # The switch of TCO method. If you want use TCO to train your model, turn it into True
    tco_switch = True if args.tco != 0 else False

    print('Training process starts:...')

    print('*' * 25)
    print("batch_size :    ", batch_size, "      |")
    print("Training Dataset: ", dataset_name, ", and Model: ", model_name)
    print("tco_switch: ", tco_switch)
    print('*' * 25)
    print('Epoch\tTrainLoss\tTrainAcc\tTestAcc')
    print('-' * 50)

    if dataset == 'cifar100':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../train_val_test'))  # get data root path
    if dataset == 'cifar100_train_val':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../fine'))  # get data root path
    if dataset == 'cifar10':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../cifar10/whole_data'))  # get data root path
    if dataset == 'cifar20':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../class20_311'))  # get data root path
    if dataset == 'ImageNet1k':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../ImageNet'))  # get data root path
    if dataset == 'cifar10_tiny':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../cifar10/cifar10_tiny'))  # get data root path
    if dataset == 'cifar311':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../cifar311'))  # get data root path
    if dataset == 'tiny_ImageNet_811':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../tiny-ImageNet200/dataset_811'))  # get data root path
    if dataset == 'tiny_ImageNet_622':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../tiny-ImageNet200/dataset_622'))  # get data root path
    if dataset == 'CUB311':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../CUB_311'))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform['val'])
    validate_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)

    # val_dataset: 用于半监督训练的验证数据集
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform['train'])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                        transform=data_transform['test'])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)

    print("using {} images for training, {} images for validation, {} images for test".format(train_num, validate_num,
                                                                                              test_num))

    for run in range(1):
        print(f"Run {run + 1}/3")

        if model_weight_path == './resnet152-pre.pth':
            net = resnet152().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.fc = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './resnet50-pre.pth':
            net = resnet50().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.fc = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './resnext50-pre.pth':
            net = resnext50_32x4d().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.fc = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './resnest50-pre.pth':
            net = resnest50().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.fc = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './efficientnet-b3.pth':
            net = EfficientNet.from_name('efficientnet-b3', num_classes=1000).to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net._fc = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './densenet121.pth':
            net = densenet121().to('cuda')
            load_state_dict(net, model_weight_path)  # for DenseNet
            net.classifier = nn.Linear(channel, num_classes).to('cuda')

        elif model_weight_path == './pvt_small.pth':
            net = pvt_small().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.head = nn.Linear(channel, num_classes).to('cuda')  # for Pvt, CvT, SwinT, poolformer, transNeXt

        elif model_weight_path == './CvT-13.pth':
            net = ConvolutionalVisionTransformer().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.head = nn.Linear(channel, num_classes).to('cuda')  # for Pvt, CvT, SwinT, poolformer, transNeXt

        elif model_weight_path == './swin_tiny_patch4_window7_224.pth':
            net = SwinTransformer().to('cuda')
            load_pretrained(model_weight_path, net)  # for swin
            net.head = nn.Linear(channel, num_classes).to('cuda')  # for Pvt, CvT, SwinT, poolformer, transNeXt

        elif model_weight_path == "./poolformer_s24.pth.tar":
            net = poolformer_s24().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.head = nn.Linear(channel, num_classes).to('cuda')  # for Pvt, CvT, SwinT, poolformer, transNeXt

        elif model_weight_path == './pvig_ti_78.5.pth.tar':
            net = pvig_ti_224_gelu().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.prediction = nn.Sequential(nn.Conv2d(384, channel, 1, bias=True),
                                      nn.BatchNorm2d(1024),
                                      act_layer('gelu'),
                                      nn.Dropout(0),
                                      nn.Conv2d(channel, num_classes, 1, bias=True)).to('cuda')  #for pVig

        elif model_weight_path == './transnext_tiny_224_1k.pth':
            net = transnext_tiny().to('cuda')
            net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
            net.head = nn.Linear(channel, num_classes).to('cuda')  # for Pvt, CvT, SwinT, poolformer, transNeXt

        # net.fc = nn.Linear(channel, num_classes).to('cuda')
        # net._fc = nn.Linear(channel, num_classes).to('cuda')      # for efficientNet
        # net.classifier = nn.Linear(channel, num_classes).to('cuda')  # for denseNet
        # net.head = nn.Linear(channel, num_classes).to('cuda')    # for Pvt, CvT, SwinT, poolformer, transNeXt
        # net.prediction = nn.Sequential(nn.Conv2d(384, channel, 1, bias=True),
        #                           nn.BatchNorm2d(1024),
        #                           act_layer('gelu'),
        #                           nn.Dropout(0),
        #                           nn.Conv2d(channel, num_classes, 1, bias=True)).to('cuda')  #for pVig

        # init.xavier_uniform_(net.fc.weight)
        # init.zeros_(net.fc.bias)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

        # feature augmentation
        num_heads = args.heads
        net.fc_q = nn.ModuleList([nn.Linear(channel, output_fc_qkv).to('cuda') for _ in range(num_heads)])
        net.bn_q = nn.ModuleList([nn.BatchNorm1d(output_fc_qkv).to('cuda') for _ in range(num_heads)])
        net.fc_k = nn.ModuleList([nn.Linear(channel, output_fc_qkv).to('cuda') for _ in range(num_heads)])
        net.bn_k = nn.ModuleList([nn.BatchNorm1d(output_fc_qkv).to('cuda') for _ in range(num_heads)])
        net.fc_v = nn.ModuleList([nn.Linear(channel, output_fc_qkv).to('cuda') for _ in range(num_heads)])
        net.bn_v = nn.ModuleList([nn.BatchNorm1d(output_fc_qkv).to('cuda') for _ in range(num_heads)])
        # net.bn_p = nn.ModuleList([nn.BatchNorm1d(channel).to('cuda') for _ in range(num_heads)])
        # net.bn_x = nn.ModuleList([nn.BatchNorm1d(channel).to('cuda') for _ in range(num_heads)])

        net.fc_q.apply(init_weights_xavier)
        net.fc_k.apply(init_weights_xavier)
        net.fc_v.apply(init_weights_xavier)

        net.fc_a = nn.Linear(num_heads * output_fc_qkv, output_fc_a).to('cuda')
        net.fc_a.apply(init_weights_kaiming)
        net.fc_r = nn.Linear(output_fc_a, channel).to('cuda')
        net.fc_r.apply(init_weights_kaiming)

        schedule = OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25,
            final_div_factor=1000
        )

        if tco_switch:
            # doubleSGD: Random Drop Optimizer transformed form SGD
            Tco_mode = Tco(net=net, pseudo_loader=val_loader, train_dataset = train_dataset, confidence=confidence, lambda_aug=args.lambda_aug)
        else:
            print("Not use TCO method in your training")

        best_acc_train = 0
        best_acc_val = 0
        best_acc_test = 0.0
        last_loss = 0.0  # 上一轮的训练loss
        tag = False
        lambda_aug = 0
        for epoch in range(epochs):
            net.train()

            num_correct = 0
            num_correct_aug = 0
            train_loss_epoch = list()
            num_total = 0
            # train_bar = tqdm(train_loader, file=sys.stdout)
            train_bar = tqdm(train_loader)
            print('training' + '*' * 25)
            train_acc = 0.0

            if epoch > args.epoch_start:
                lambda_aug = args.lambda_aug
            # 正常分类器训练
            for data in train_bar:
                imgs, labels = data
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
                optimizer.zero_grad()

                net.aug = False
                logits, _ = net(imgs.to('cuda'))

                net.aug = True
                _, pred = torch.max(logits, dim=1)
                logits_aug, _ = net(imgs.to('cuda'),pred=pred)

                log_probs = F.log_softmax(logits, dim=1)
                probs_aug = F.softmax(logits_aug, dim=1)

                # loss = criterion(logits, labels.to('cuda')) + criterion(logits_aug, labels.to('cuda')) * lambda_aug # + loss on feature augmentation
                if tco_switch:
                    loss = criterion(logits, labels.to('cuda')) + (criterion(logits_aug, labels.to('cuda')) + F.kl_div(log_probs, probs_aug, reduction='batchmean')) * lambda_aug  # + loss on feature augmentation
                    # loss = criterion(logits, labels.to('cuda'))
                    # loss = criterion(logits, labels.to('cuda')) + criterion(logits_aug, labels.to('cuda'))*lambda_aug
                    # loss = F.kl_div(log_probs, probs_aug, reduction='batchmean')
                    # loss = criterion(logits, labels.to('cuda')) + F.kl_div(log_probs, probs_aug, reduction='batchmean')
                else:
                    loss = criterion(logits, labels.to('cuda'))
                # loss = criterion(logits_aug, labels.to('cuda'))
                # loss = criterion(logits, labels.to('cuda')) + F.kl_div(log_probs, probs_aug, reduction='batchmean') * lambda_aug  # + loss on feature augmentation

                # _, pred = torch.max(logits, dim=1)
                _, pred_aug = torch.max(probs_aug, dim=1)

                num_correct += torch.sum(pred == labels.detach_())
                num_correct_aug += torch.sum(pred_aug == labels.detach_())
                num_total += labels.size(0)

                # _, pred_aug = torch.max(logits_aug, 1)
                # num_correct_aug += torch.sum(pred_aug == labels.detach_())

                train_loss_epoch.append(loss.item())
                loss.backward()
                optimizer.step()
                schedule.step()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            train_acc_epoch = num_correct.detach().cpu().numpy() * 100 / num_total
            train_acc_epoch_aug = num_correct_aug.detach().cpu().numpy() * 100 / num_total
            avg_train_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch)

            val_acc_epoch, val_acc_epoch_aug, val_acc5 = _accuracy_(net, validate_loader)
            test_accuracy, test_accuracy_aug, test_acc5 = _accuracy_(net, test_loader)
            is_best = val_acc_epoch >= best_acc_val
            if is_best:
                print("||-|-||This result is better||-|-||")
                # torch.save(net.state_dict(), save_path)
                # print('this model is saved!')

            best_acc_train = max(train_acc_epoch, best_acc_train)
            best_acc_val = max(val_acc_epoch, best_acc_val)
            if best_acc_val == val_acc_epoch:
                best_acc_test = test_accuracy

# 保留当前模型参数，回溯点
            model_checkpoint = net.state_dict()
            optimizer_checkpoint = optimizer.state_dict()
            scheduler_checkpoint = schedule.state_dict()

            if tco_switch and ((best_acc_val > threshold_acc) or epoch>args.epoch_start):  # 判断模型是否收敛？
                print(
                    'before TCO: \tval_acc: {:.2f}%\ttest_acc: {:.2f}\tval_acc_aug: {:.2f}\ttest_acc_aug: {:.2f}\tval_acc5: {:.2f}\ttest_acc5: {:.2f}'.format(val_acc_epoch, test_accuracy, val_acc_epoch_aug, test_accuracy_aug, val_acc5, test_acc5))
                current_lr = optimizer.param_groups[0]['lr']
                print('the current lr is:{}'.format(current_lr))
                print('using TCO!!!' + '***' * 10)
                net, pseudo_output, trully_output = Tco_mode(net, batch_size, optimizer=optimizer, nw=nw)
                # torch.save(pseudo_output,'./labels_output/no_CB/epoch_{}_pseudo.pt'.format(epoch))
                # torch.save(trully_output, './labels_output/no_CB/epoch_{}_trully.pt'.format(epoch))
                # torch.save(pseudo_output, './labels_output/CB/epoch_{}_pseudo.pt'.format(epoch))
                # torch.save(trully_output, './labels_output/CB/epoch_{}_trully.pt'.format(epoch))
                # torch.save(pseudo_output, './labels_output/no_aug/epoch_{}_pseudo.pt'.format(epoch))
                # torch.save(trully_output, './labels_output/no_aug/epoch_{}_trully.pt'.format(epoch))
                # torch.save(pseudo_output, './labels_output/aug/epoch_{}_pseudo.pt'.format(epoch))
                # torch.save(trully_output, './labels_output/aug/epoch_{}_trully.pt'.format(epoch))

            tco_val, tco_val_aug, val_acc5 = _accuracy_(net, validate_loader)
            tco_test, tco_test_aug, test_acc5= _accuracy_(net, test_loader)

            val_loss = _loss_(net, validate_loader)
            # torch.save(train_loss_epoch, './labels_output/no_rewind/epoch_{}_train.pt'.format(epoch))
            # torch.save(val_loss, './labels_output/no_rewind/epoch_{}_val.pt'.format(epoch))
            # torch.save(train_loss_epoch, './labels_output/rewind/epoch_{}_train.pt'.format(epoch))
            # torch.save(val_loss, './labels_output/rewind/epoch_{}_val.pt'.format(epoch))
            print('The accuracy on test set is: {}'.format(tco_test))

            is_best = tco_val >= best_acc_val
            if is_best:
                # print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch + 1, avg_train_loss_epoch, train_acc_epoch, val_acc_epoch),end='')
                print("||-|-||This result is better||-|-||")
                torch.save(net.state_dict(), './weights/baseline.pth')
                # torch.save(net.state_dict(), './weights/self-learning.pth')
                print('this model is saved!')
            print(
                'epoch: {}\ttrain_loss: {:.4f}\ttrain_acc: {:.2f}%\ttrain_aug_acc: {:.2f}%\tval_acc: {:.2f}%\ttest_acc: {:.2f}\tval_acc_aug: {:.2f}%\ttest_acc_aug: {:.2f}\tval_acc5: {:.2f}\ttest_acc5: {:.2f}'.format(
                    epoch + 1, avg_train_loss_epoch,
                    train_acc_epoch, train_acc_epoch_aug, tco_val, tco_test, tco_val_aug, tco_test_aug, val_acc5, test_acc5))
            best_acc_train = max(train_acc_epoch, best_acc_train)
            best_acc_val = max(tco_val, best_acc_val)
            if best_acc_val == tco_val:
                best_acc_test = tco_test
                best_acc5_val = val_acc5
                best_acc5_test = test_acc5

#####   rewind
            if tco_val < val_acc_epoch - args.rewind_threshold:
                net.load_state_dict(model_checkpoint)
                optimizer.load_state_dict(optimizer_checkpoint)
                schedule.load_state_dict(scheduler_checkpoint)
                print(
                    "--------------------------------------The model weights are backtracked-------------------------------------")

        print('Run {} - best_acc_train:{:.2f}\tbest_acc_val:{:.2f}%\tbest_acc_test:{:.2f}%\tbest_acc5_val:{:.2f}%\tbest_acc5_test:{:.2f}%'.format(run + 1,
                                                                                                   best_acc_train,
                                                                                                   best_acc_val,
                                                                                                   best_acc_test,
                                                                                                    best_acc5_val,
                                                                                                    best_acc5_test))
    print("Training process finished!!!")

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # 确保使用 spawn 方法
    main()