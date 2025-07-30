import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from torchvision import transforms, datasets
from model import resnet50
import time
from torch.optim.lr_scheduler import OneCycleLR
# current_dir = os.path.dirname(os.path.abspath(__file__))# 获取当前文件的目录
# project_root = os.path.dirname(current_dir)# 获取项目的根目录
# sys.path.append(project_root)# 将项目根目录添加到 sys.path
from lql_method import Tco
# from faker_method import Tco
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--nw", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_confidence", type=float, default=0.999)
parser.add_argument("--min_confidence", type=float, default=0.99)
parser.add_argument("--tco", type=int, default=1)
parser.add_argument("--threshold_acc", type=int, default=0)
parser.add_argument("--margin", type=float, default=0.3)
parser.add_argument("--tag", type=str, choices=['max', 'margin'],
                    help='choose one of the followed options: max, margin', default='max')
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--dataset", type=str, default='cifar10_tiny')
parser.add_argument("--topN", type=int, default=20)
parser.add_argument("--confidence", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--length_top", type=float, default=50)
args = parser.parse_args()
print('tag:{}'.format(args.tag))
print('max_confidence:{}'.format(args.max_confidence))
print('min_confidence:{}'.format(args.min_confidence))
print('margin:{}'.format(args.margin))
print('acc:{}'.format(args.threshold_acc))
print('alpha:{}'.format(args.alpha))
print('confidence:{}'.format(args.confidence))
print('lr:{}'.format(args.lr))


def main():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    def _accuracy(net, data_loader, str):
        net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            # if str == 'validate':
            #     print('validating' + '*' * 25)
            # if str == 'test':
            #     print('testing' + '*' * 25)
            loader_bar = tqdm(data_loader, file=sys.stdout)
            for imgs, labels in data_loader:
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
                output = net(imgs)
                _, pred = torch.max(output, 1)
                num_acc += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
        LV = num_acc.detach().cpu().numpy() * 100 / num_total
        return LV

    epochs = args.epochs
    base_lr = args.lr
    momentum = 0.9
    weight_decay = args.weight_decay
    step_size = epochs / 2
    gamma = 0.1
    batch_size = args.batch_size
    nw = args.nw
    model_name = 'Res50'
    dataset_name = args.dataset
    save_path = 'weights/' + model_name + '.pth'
    model_weight_path = 'resnet50-pre.pth'
    channel = 2048
    num_classes = 10

    # The confidence threshold
    max_confidence = args.max_confidence
    min_confidence = args.min_confidence
    threshold_acc = args.threshold_acc
    margin = args.margin
    alpha = args.alpha
    dataset = args.dataset
    topN = args.topN
    confidence = args.confidence

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
    if dataset == 'ImageNet1k':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../ImageNet'))  # get data root path
    if dataset == 'cifar10_tiny':
        data_root = os.path.abspath(os.path.join(os.getcwd(), '../../cifar10/cifar10_tiny'))  # get data root path
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
    val_num = len(val_dataset)
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

    for run in range(3):
        print(f"Run {run + 1}/3")
        net = resnet50().to('cuda')
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        net.fc = nn.Linear(channel, num_classes).to('cuda')
        net.bce = False
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, weight_decay=weight_decay, momentum=momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay)
        # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
            Tco_mode = Tco(net=net, validate_loader=val_loader, max_confidence=max_confidence,
                           min_confidence=min_confidence, margin=margin, lr=base_lr, tag=args.tag, confidence=confidence)
        else:
            print("Not use TCO method in your training")

        best_acc_train = 0
        best_acc_val = 0
        best_acc_test = 0.0
        last_acc = 0.0  # 上一个模型权重的验证精度
        last_loss = 0.0  # 上一轮的训练loss
        count_backtrack = 0
        tag = False
        current_step = 0
        total_steps = epochs * len(train_loader)
        print(total_steps)
        for epoch in range(epochs):
            net.train()
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            train_bar = tqdm(train_loader, file=sys.stdout)
            print('training' + '*' * 25)
            for data in train_bar:
                imgs, labels = data
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
                output = net(imgs)
                loss = criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if tco_switch:
                    if Tco_mode.current_step < total_steps:
                        schedule.step()         # OneCycleLR
                        Tco_mode.current_step += 1
                else:
                    schedule.step()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            train_acc_epoch = num_correct.detach().cpu().numpy() * 100 / num_total
            avg_train_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch)

            val_acc_epoch = _accuracy(net, validate_loader, 'validate')
            test_accuracy = _accuracy(net, test_loader, 'test')
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


            if tco_switch and ((best_acc_val > threshold_acc) or (epoch > epochs / 2 and avg_train_loss_epoch > last_loss) or tag):  # 判断模型是否收敛？
                tag = True
                print(
                    'before TCO: \tval_acc: {:.2f}%\ttest_acc: {:.2f}'.format(val_acc_epoch, test_accuracy))
                allocated = [val_num / num_classes] * num_classes
                current_lr = optimizer.param_groups[0]['lr']
                print('the current lr is:{}'.format(current_lr))
                print('using TCO!!!' + '***' * 10)
                net = Tco_mode(net, current_lr, alpha, allocated, batch_size, topN=topN, optimizer=optimizer, scheduler=schedule, current_step=current_step, total_steps=total_steps)

            tco_val = _accuracy(net, validate_loader, 'validate')
            tco_test = _accuracy(net, test_loader, 'test')
            print('The accuracy on test set is: {}'.format(tco_test))
            # schedule.step() # 一般的调度器
            is_best = tco_val >= best_acc_val
            if is_best:
                # print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%'.format(epoch + 1, avg_train_loss_epoch, train_acc_epoch, val_acc_epoch),end='')
                print("||-|-||This result is better||-|-||")
                # torch.save(net.state_dict(), save_path)
                # print('this model is saved!')
            print(
                'epoch: {}\ttrain_loss: {:.4f}\ttrain_acc: {:.2f}%\tval_acc: {:.2f}%\ttest_acc: {:.2f}'.format(
                    epoch + 1, avg_train_loss_epoch,
                    train_acc_epoch, tco_val, tco_test))
            best_acc_train = max(train_acc_epoch, best_acc_train)
            best_acc_val = max(tco_val, best_acc_val)
            if best_acc_val == tco_val:
                best_acc_test = tco_test

            if tco_val < val_acc_epoch - 5:
                net.load_state_dict(model_checkpoint)
                optimizer.load_state_dict(optimizer_checkpoint)
                schedule.load_state_dict(scheduler_checkpoint)
                print(
                    "--------------------------------------The model weights are backtracked-------------------------------------")

        print('Run {} - best_acc_train:{:.4f}\tbest_acc_val:{:.4f}%\tbest_acc_test:{:.4f}%'.format(run + 1,
                                                                                                   best_acc_train,
                                                                                                   best_acc_val,
                                                                                                   best_acc_test))
    print("Training process finished!!!")


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # 确保使用 spawn 方法
    main()