import torch
import torch.nn as nn
from layers import *
from data import sixray
import os
import torch.nn.functional as F

class SSD(nn.Module):
    '''
     Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    '''
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = sixray
        self.priorbox = PriorBox(self.cfg)#layers/functions/prior_box.py class PriorBox(object)
        self.priors = self.priorbox.forward()
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)# layers/modules/l2norm.py class L2Norm(nn.Module)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])# head = (loc_layers, conf_layers)
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.softmax(dim=1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)#  layers/functions/detection.py class Detect
            # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.

    def forward(self, x):
        '''
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300]?.
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        '''
        sources = list()#参与预测的6个卷积层输出
        loc = list()#存储预测边框信息
        conf = list()#存储预测类别信息

        #计算vgg直到conv4_3的relu
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        #conv7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(s)

        #extra_layers
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        #apply multibox head to source layers
        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 (N, H, W, C).
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]???
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) 
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 最终的shape为(两维):[batch, num_boxes*num_classes]

        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),    # [batch, num_boxes, 4], [1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors                      # 利用 PriorBox对象获取特征图谱上的 default box, 该参数的shape为: [8732,4]   
            )
        if self.phase == 'train':
            output = (
                loc.view(loc.size(0), -1, 4), 
                conf.view(conf.size(0), -1, self.num_classes), 
                self.priors
            )

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            # Load all tensors onto the CPU, using a function
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py    
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):#for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]   #21:conv4_3, -2:conv7
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]#vgg16,conv1-5
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]#conv8-conv11
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4]
}

def build_ssd(phase, size=300, num_classes=3):
    if pahse != 'test' and phase != 'train':
        print('ERROR: Phase: ' + phase + 'not recognized')
        return
    '''
    if size != 300:
        print('ERROR: You specified size ' + repr(size) + '. However, ' + \
            'currently only SSD300 (size=300) is supperted!')
        return
    '''
    base_, extras_, head_ = multibox(
        vgg(base[str(size)], 3),#in_channels=3,RGB输入3channels
        add_extras(extras[str(size)], 1024),
        mbox[str(size)],
        num_classes
    )#head:(loc_layers, conf_layers)
    return SSD(phase, size, base_, extras_, head_, num_classes)    