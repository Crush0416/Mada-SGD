'''
---------backbone network-------
-----------resnet4-----------
# awaitting to further progress and test
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out

def ConvBlockFunction(x, w1, b1, w_bn1, b_bn1, w2, b2, w_bn2, b_bn2):
    out = F.conv2d(x, w1, b1, padding=1)
    out = F.batch_norm(out, running_mean=None, running_var=None, weight=w_bn1, bias=b_bn1, training=True)
    out = F.relu(out)
    out = F.max_pool2d(out, kernel_size=2, stride=2)
    out = F.conv2d(out, w2, b2, padding=1)
    out = F.batch_norm(out, running_mean=None, running_var=None, weight=w_bn2, bias=b_bn2, training=True)
    
    return out
        
        
class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        return out

def DownSampleFunction(x, w, b, w_bn, b_bn):
    out = F.conv2d(x, w, b, stride=2)
    out = F.batch_norm(out, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
        
    return out    
        

class ResNet8(nn.Module):
    def __init__(self, in_channels, n_way, out_channels=64):
        super(ResNet8, self).__init__()
        self.layer1 = ConvBlock(in_channels, out_channels)
        self.layer2 = ConvBlock(out_channels, out_channels)
        self.layer3 = ConvBlock(out_channels, out_channels)
        self.layer4 = ConvBlock(out_channels, out_channels)                     
        
        self.downsample1 = DownSample(in_channels, out_channels)
        self.downsample2 = DownSample(out_channels, out_channels)
        self.downsample3 = DownSample(out_channels, out_channels)
        self.downsample4 = DownSample(out_channels, out_channels)
        
        self.fc = nn.Linear(1024, n_way)
        # self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        
        residule = self.downsample1(x)
        out = self.layer1(x)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample2(out)
        out = self.layer2(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample3(out)
        out = self.layer3(out)
        out += residule
        out = F.relu(out)
        
        residule = self.downsample4(out)
        out = self.layer4(out)
        out += residule
        out = F.relu(out)
        
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        # out = self.softmax(out)

        return out

    def functional_forward(self, x, params):
        residule = DownSampleFunction(x, params[f'downsample1.conv.weight'], params[f'downsample1.conv.bias'],
                                         params.get(f'downsample1.bn.weight'), params.get(f'downsample1.bn.bias') )               
        out = ConvBlockFunction(x, params[f'layer1.conv1.weight'], params[f'layer1.conv1.bias'],
                                   params.get(f'layer1.bn1.weight'), params.get(f'layer1.bn1.bias'),
                                   params[f'layer1.conv2.weight'], params[f'layer1.conv2.bias'],
                                   params.get(f'layer1.bn2.weight'), params.get(f'layer1.bn2.bias'))
        out += residule
        out = F.relu(out)

        residule = DownSampleFunction(out, params[f'downsample2.conv.weight'], params[f'downsample2.conv.bias'],
                                         params.get(f'downsample2.bn.weight'), params.get(f'downsample2.bn.bias') )               
        out = ConvBlockFunction(out, params[f'layer2.conv1.weight'], params[f'layer2.conv1.bias'],
                                   params.get(f'layer2.bn1.weight'), params.get(f'layer2.bn1.bias'),
                                   params[f'layer2.conv2.weight'], params[f'layer2.conv2.bias'],
                                   params.get(f'layer2.bn2.weight'), params.get(f'layer2.bn2.bias'))
        out += residule
        out = F.relu(out)                      

        residule = DownSampleFunction(out, params[f'downsample3.conv.weight'], params[f'downsample3.conv.bias'],
                                         params.get(f'downsample3.bn.weight'), params.get(f'downsample3.bn.bias') )               
        out = ConvBlockFunction(out, params[f'layer3.conv1.weight'], params[f'layer3.conv1.bias'],
                                   params.get(f'layer3.bn1.weight'), params.get(f'layer3.bn1.bias'),
                                   params[f'layer3.conv2.weight'], params[f'layer3.conv2.bias'],
                                   params.get(f'layer3.bn2.weight'), params.get(f'layer3.bn2.bias'))
        out += residule
        out = F.relu(out)
        
        residule = DownSampleFunction(out, params[f'downsample4.conv.weight'], params[f'downsample4.conv.bias'],
                                         params.get(f'downsample4.bn.weight'), params.get(f'downsample4.bn.bias') )               
        out = ConvBlockFunction(out, params[f'layer4.conv1.weight'], params[f'layer4.conv1.bias'],
                                   params.get(f'layer4.bn1.weight'), params.get(f'layer4.bn1.bias'),
                                   params[f'layer4.conv2.weight'], params[f'layer4.conv2.bias'],
                                   params.get(f'layer4.bn2.weight'), params.get(f'layer4.bn2.bias'))
        out += residule
        out = F.relu(out)        

        out = out.view(x.shape[0], -1)
        out = F.linear(out, params['fc.weight'], params['fc.bias'])
        # out = F.softmax(out)

        return out