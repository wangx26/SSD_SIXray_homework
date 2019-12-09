import os
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

SIXRAY_CLASSES = ('带电芯充电宝', '不带电芯充电宝')
SIXRAY_ROOT = osp.join(HOME, 'data/sixray/')

def get_label_map(label_file):
    label_map = {}
    with open(label_file, 'r') as labels:
        for line in labels:
            ids = line.split(',')
            label_map[int(ids[0])] = int(ids[1])
    return label_map

class SIXRAYAnnotationTransform(object):
    """Transforms a annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(SIXRAY_CLASSES, range(len(SIXRAY_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height, idx):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable will be an ET.Element            
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """       
        # 遍历Annotation
        res = []
        with open(target, "r", encoding='utf-8') as f1:
            dataread = f1.readlines()
        for annotation in dataread:
            if not annotation:
                continue
            bndbox = []
            temp = annotation.split()
            name = temp[1]
            # 只读两类
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2]) / width
            # 只读取V视角的
            if xmin > 1:
                continue
            if xmin < 0:
                xmin = 0
            ymin = int(temp[3]) / height
            if ymin < 0:
                ymin = 0
            xmax = int(temp[4]) / width
            if xmax > 1:
                xmax = 1
            ymax = int(temp[5]) / height
            if ymax > 1:
                ymax = 1
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            label_idx = self.class_to_ind[name]
            # label_idx = name
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        if len(res) == 0:
            return [[0, 0, 0, 0, 3]]
        return res

class SIXRAYDetection(data.Dataset):
    """Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
    """
    def __init__(self, root, imagesets=osp.join(SIXRAY_ROOT, 'train.txt'), transform=None, target_transform=SIXRAYAnnotationTransform(),\
         dataset_name='SIXRAY'):
        self.root = root
        self.image_set = imagesets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s' % self.root, 'core_3000/Annotation/core_battery%s.txt')
        self._imgpath = osp.join('%s' % self.root, 'core_3000/Image/core_battery%s.jpg')
        self._annopath1 = osp.join('%s' % self.root, 'coreless_3000/Annotation/coreless_battery%s.txt')
        self._imgpath1 = osp.join('%s' % self.root, 'coreless_3000/Image/coreless_battery%s.jpg')
        self.ids = list()

        with open(self.image_set, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.ids.append(line.strip('\n'))

    def __getitem__(self, index):
        im, gt, h, w, og_im = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = self._annopath % img_id  # 注释目录
        # print(target)
        # print(self._imgpath % img_id)
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            target = self._annopath1 % img_id
            img = cv2.imread(self._imgpath1 % img_id)        
        if img is None:
            print('\nwrong\n')
            print(self._imgpath1 % img_id)
        height, width, channels = img.shape
        # print("height: " + str(height) + " ; width : " + str(width) + " ; channels " + str(channels) )
        og_img = img

        # print (img_id)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height, img_id)
            target = np.array(target)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(a2, 0, a1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, og_img
        # return torch.from_numpy(img), target, height, width