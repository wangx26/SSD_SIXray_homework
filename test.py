from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import SIXRAY_ROOT, SIXRAY_CLASSES as labelmap
from PIL import Image
from data import SIXRAYAnnotationTransform, SIXRAYDetection, BaseTransform, SIXRAY_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd512_sixray_72000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='../predicted_file/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.01, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--sixray_root', default=SIXRAY_ROOT, help='Location of SIXRAY root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename1 = save_folder+'det_test_带电芯充电宝.txt'
    filename2 = save_folder+'det_test_不带电芯充电宝.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img, type_core, img_id = testset.pull_image(i)
        _, annotation, height, width, og_img = testset.pull_item(i)
        #img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0).type(torch.FloatTensor)
          
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data 
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                if label_name == '带电芯充电宝':
                    with open(filename1, mode='a') as f:
                        f.write(type_core + img_id + ' ' + str(float(score)) + ' '+' '.join(str(c) for c in coords) + '\n')                        
                elif label_name == '不带电芯充电宝':
                    with open(filename2, mode='a') as f:
                        f.write(type_core + img_id + ' ' + str(float(score)) + ' '+' '.join(str(c) for c in coords) + '\n')
                j += 1


def test(img_path, anno_path):
    # load net
    num_classes = len(SIXRAY_CLASSES) + 1 # +1 background
    net = build_ssd('test', 512, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = SIXRAYDetection(args.sixray_root, img_path=img_path, anno_path=anno_path, target_transform=SIXRAYAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (211, 221, 225)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test('/home/wangxu/Projects/SSD/data/sixray/Image_test', '/home/wangxu/Projects/SSD/data/sixray/Anno_test')
