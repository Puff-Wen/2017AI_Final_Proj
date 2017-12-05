'''Draw the result from loss and acc files'''
''' This is a test '''
from __future__ import print_function

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

parser = argparse.ArgumentParser(description='Draw the result from loss and acc files')
parser.add_argument('--net', '-n', type=str, required=True)
args = parser.parse_args()
print('==> The net is..', args.net)

print('==> Start to draw loss')
# Read data from files
train_loss_list20 = [line.rstrip('\n') for line in open(args.net + "20_train_loss.txt", "r")]
train_loss_list56 = [line.rstrip('\n') for line in open(args.net + "56_train_loss.txt", "r")]
train_loss_list110 = [line.rstrip('\n') for line in open(args.net + "110_train_loss.txt", "r")]
test_loss_list20 = [line.rstrip('\n') for line in open(args.net + "20_test_loss.txt", "r")]
test_loss_list56 = [line.rstrip('\n') for line in open(args.net + "56_test_loss.txt", "r")]
test_loss_list110 = [line.rstrip('\n') for line in open(args.net + "110_test_loss.txt", "r")]

# convert to float
train_loss_list20 = map(float, train_loss_list20)
train_loss_list56 = map(float, train_loss_list56)
train_loss_list110 = map(float, train_loss_list110)
test_loss_list20 = map(float, test_loss_list20)
test_loss_list56 = map(float, test_loss_list56)
test_loss_list110 = map(float, test_loss_list110)

fig = plt.figure()
fig.suptitle(args.net+'_loss', fontsize=20)
x = np.arange(len(train_loss_list20))

plt.subplot(121)
plt.plot(x, train_loss_list20)
plt.plot(x, train_loss_list20, 'rx', ms=3, label=args.net+'20')
plt.plot(x, train_loss_list56)
plt.plot(x, train_loss_list56, 'go', ms=3, label=args.net+'56')
plt.plot(x, train_loss_list110)
plt.plot(x, train_loss_list110, 'b^', ms=3, label=args.net+'110')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('train loss', fontsize=12)
plt.ylim([0, 2.5])
plt.legend(shadow=True)

plt.subplot(122)
plt.plot(x, test_loss_list20)
plt.plot(x, test_loss_list20, 'rx', ms=3, label=args.net+'20')
plt.plot(x, test_loss_list56)
plt.plot(x, test_loss_list56, 'go', ms=3, label=args.net+'56')
plt.plot(x, test_loss_list110)
plt.plot(x, test_loss_list110, 'b^', ms=3, label=args.net+'110')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('test loss', fontsize=12)
plt.ylim([0, 2.5])
plt.legend(shadow=True)

plt.subplots_adjust(top=0.85, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

fig.savefig(args.net+'_loss.png')
print('<== Complete to draw loss')

print('==> Start to draw acc')
# Read data from files
train_acc_list20 = [line.rstrip('\n') for line in open(args.net + "20_train_acc.txt", "r")]
train_acc_list56 = [line.rstrip('\n') for line in open(args.net + "56_train_acc.txt", "r")]
train_acc_list110 = [line.rstrip('\n') for line in open(args.net + "110_train_acc.txt", "r")]
test_acc_list20 = [line.rstrip('\n') for line in open(args.net + "20_test_acc.txt", "r")]
test_acc_list56 = [line.rstrip('\n') for line in open(args.net + "56_test_acc.txt", "r")]
test_acc_list110 = [line.rstrip('\n') for line in open(args.net + "110_test_acc.txt", "r")]
print('====> The best acc of train_acc_list20 is', max(train_acc_list20))
print('====> The best acc of train_acc_list56 is', max(train_acc_list56))
print('====> The best acc of train_acc_list110 is', max(train_acc_list110))
print('====> The best acc of test_acc_list20 is', max(test_acc_list20))
print('====> The best acc of test_acc_list56 is', max(test_acc_list56))
print('====> The best acc of test_acc_list110 is', max(test_acc_list110))

# convert to float
train_acc_list20 = map(float, train_acc_list20)
train_acc_list56 = map(float, train_acc_list56)
train_acc_list110 = map(float, train_acc_list110)
test_acc_list20 = map(float, test_acc_list20)
test_acc_list56 = map(float, test_acc_list56)
test_acc_list110 = map(float, test_acc_list110)

fig = plt.figure()
fig.suptitle(args.net+'_acc', fontsize=20)
x = np.arange(len(train_acc_list20))

plt.subplot(121)
plt.plot(x, train_acc_list20)
plt.plot(x, train_acc_list20, 'rx', ms=3, label=args.net+'20')
plt.plot(x, train_acc_list56)
plt.plot(x, train_acc_list56, 'go', ms=3, label=args.net+'56')
plt.plot(x, train_acc_list110)
plt.plot(x, train_acc_list110, 'b^', ms=3, label=args.net+'110')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('train acc(%)', fontsize=12)
plt.ylim([0, 100])
plt.legend(shadow=True)

plt.subplot(122)
plt.plot(x, test_acc_list20)
plt.plot(x, test_acc_list20, 'rx', ms=3, label=args.net+'20')
plt.plot(x, test_acc_list56)
plt.plot(x, test_acc_list56, 'go', ms=3, label=args.net+'56')
plt.plot(x, test_acc_list110)
plt.plot(x, test_acc_list110, 'b^', ms=3, label=args.net+'110')
plt.xlabel('epoch', fontsize=12)
plt.ylabel('test acc(%)', fontsize=12)
plt.ylim([0, 100])
plt.legend(shadow=True)

plt.subplots_adjust(top=0.85, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

fig.savefig(args.net+'_acc.png')
print('<== Complete to draw acc')