'''
    -------*- coding: utf-8 -*-------
    -------backbone architecture-----
    -------------Conv4---------------
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


def ConvBlockFunction(input, w, b, w_bn, b_bn):
    x = F.conv2d(input, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)

    return output


class ConvNet4(nn.Module):
    def __init__(self, in_ch, n_way):
        super(ConvNet4, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(1024, n_way)    # 64*64-1024, 80*80-1600, 96*96-2304, 112*112-3136, 128*128-4096
        # self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        # x = self.softmax(x)

        return x

    def functional_forward(self, x, params):
        x = ConvBlockFunction(x, params[f'conv1.conv2d.weight'], params[f'conv1.conv2d.bias'],
                              params.get(f'conv1.bn.weight'), params.get(f'conv1.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv2.conv2d.weight'], params[f'conv2.conv2d.bias'],
                              params.get(f'conv2.bn.weight'), params.get(f'conv2.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv3.conv2d.weight'], params[f'conv3.conv2d.bias'],
                              params.get(f'conv3.bn.weight'), params.get(f'conv3.bn.bias'))
        x = ConvBlockFunction(x, params[f'conv4.conv2d.weight'], params[f'conv4.conv2d.bias'],
                              params.get(f'conv4.bn.weight'), params.get(f'conv4.bn.bias'))

        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        # x = F.softmax(x)

        return x