'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import *
from torch.autograd import Variable

import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=164, metavar='N', help='number of epochs to train (default: 164)')
parser.add_argument('--net', '-n', type=str, required=True)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data container
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616)),	
])

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..', args.net)
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
	# ######## PuffResNet ########
    # net = PuffRessNet20()
    # net = PuffResNet56()
    # net = PuffResNet110()
	# ######## PuffPreActResNet ########
    # net = PuffPreActResNet20()
    # net = PuffPreActResNet56()
    # net = PuffPreActResNet110()
    if args.net == 'PuffResNet20':
        net = PuffResNet20()
    elif args.net == 'PuffResNet56':
        net = PuffResNet56()
    elif args.net == 'PuffResNet110':
        net = PuffResNet110()	
    elif args.net == 'PuffPreActResNet20':
        net = PuffPreActResNet20()
    elif args.net == 'PuffPreActResNet56':
        net = PuffPreActResNet56()
    elif args.net == 'PuffPreActResNet110':
        net = PuffPreActResNet110()
    elif args.net == 'PuffVanillaCNN20':
        net = PuffVanillaCNN20()
    elif args.net == 'PuffVanillaCNN56':
        net = PuffVanillaCNN56()
    elif args.net == 'PuffVanillaCNN110':
        net = PuffVanillaCNN110()
    else:
        sys.exit('Not available..'+ args.net)
		
print(net)
net.apply(init_params)


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

#learning rate scheduling
def adjust_learning_rate(optimizer, epoch):

    if epoch < 81:
       lr = 0.1
    elif epoch < 122:
       lr = 0.01
    else: 
       lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#learning rate scheduling
def adjust_slow_learning_rate(optimizer, epoch):

    if epoch < 81:
       lr = 0.01
    elif epoch < 122:
       lr = 0.001
    else: 
       lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
		
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    if args.net == 'PuffVanillaCNN110':
        adjust_slow_learning_rate(optimizer, epoch)
    elif args.net == 'PuffPreActResNet20':
        adjust_slow_learning_rate(optimizer, epoch)
    elif args.net == 'PuffPreActResNet56':
        adjust_slow_learning_rate(optimizer, epoch)
    elif args.net == 'PuffPreActResNet110':
        adjust_slow_learning_rate(optimizer, epoch)
    else:
        adjust_learning_rate(optimizer, epoch)    
    train_loss = 0
    correct = 0
    total = 0
    train_loss_rate = 0.0
    train_acc_rate = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        train_loss_rate = train_loss/(batch_idx+1)
        train_acc_rate = 100.*correct/total
    train_loss_list.append(format(train_loss_rate, ".3f"))
    train_acc_list.append(format(train_acc_rate, ".3f"))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_rate = 0.0
    test_acc_rate = 0.0	
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_loss_rate = test_loss/(batch_idx+1)
        test_acc_rate = 100.*correct/total
    test_loss_list.append(format(test_loss_rate, ".3f"))
    test_acc_list.append(format(test_acc_rate, ".3f"))
	
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

time_a = datetime.datetime.now()
print("Time is ", time_a)

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)

# Write acc and loss to files
outfile1 = open(args.net + "_train_loss.txt", "w")
outfile2 = open(args.net + "_train_acc.txt", "w")
outfile3 = open(args.net + "_test_loss.txt", "w")
outfile4 = open(args.net + "_test_acc.txt", "w")
outfile5 = open(args.net + "_exe_time.txt", "w")
for i in range(start_epoch, start_epoch+args.epochs):
    outfile1.write(train_loss_list[i])
    outfile1.write("\n")
    outfile2.write(train_acc_list[i])
    outfile2.write("\n")
    outfile3.write(test_loss_list[i])
    outfile3.write("\n")
    outfile4.write(test_acc_list[i])
    outfile4.write("\n")


print("best_acc is ", best_acc)
time_b = datetime.datetime.now()
print("Time is ", time_b)
print("Total computation time: ", time_b - time_a)
outfile5.write("Total computation time: " + str(time_b - time_a))

outfile1.close()
outfile2.close()
outfile3.close()
outfile4.close()
outfile5.close()