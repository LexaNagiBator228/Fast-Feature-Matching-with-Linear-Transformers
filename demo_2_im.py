import argparse
import glob
import numpy as np
import os
import time
import torch.utils.data
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import json
import torch.nn.functional as F
import time
import cv2 as cv
import torch

import argparse

from model import Linear_net_small
from PIL import Image
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from models_S.superpoint import SuperPoint


def draw_interest_points_rgb(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = img
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb

def draw_interest_points(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb


def draw_lines(img, p1,p2):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(p1.shape[0]):
        cv.line(img_rgb, (p1[i][0], p1[i][1]), (p2[i][0]+ 512, p2[i][1] ),  (205, 0, 0), 1)
        cv.circle(img_rgb, (p1[i][0], p1[i][1]), 2, (0, 255, 0), -1)
        cv.circle(img_rgb, (p2[i][0] +512, p2[i][1]), 2, (0, 255, 0), -1)
    return img_rgb

print("hi")
print('start')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

default_config = {
        'descriptor_dim': 256,
        'nms_radius': 3,
        'keypoint_threshold': 0.05,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

net = SuperPoint(default_config).to(device)
from models_S.matching import Matching


parser = argparse.ArgumentParser()
parser.add_argument('--image1_path', type=str,
                        help='The image path of the testing image', default='./media/im1.jpg')
parser.add_argument('--image2_path', type=str,
                        help='The path to the second image', default='./media/im2.jpg' ) 
parser.add_argument("-v","--vis", 
                        help='Whether to show results', action="store_true") 

parser.add_argument('--save_image', type=bool,
                        help='Whether to save the results', default=True)         
parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./weights/model_new_temp40.pth')         

args = parser.parse_args()
print(args)



f_net  = Linear_net_small(4,256,8,256,256,4).to(device)


#path_m = 'model_new_temp40.pth'
path_m = args.model_dir
params = torch.load(path_m, map_location= device)
f_net.load_state_dict(torch.load(path_m, map_location=device))
net.eval()
f_net.eval()





with torch.no_grad():
    im1_path = args.image1_path
    im2_path = args.image2_path
    im1 = cv2.resize(cv2.imread(im1_path, 0), (512,512))


    gray = cv2.resize(cv2.imread(im2_path, 0), (512,512))
    im2 = gray
    m1 = im1
    m2 = im2


    im1 = torch.from_numpy(im1).float().to(device).float()/255.
    im2 = torch.from_numpy(im2).float().to(device).float()/255.

    im1 = im1.view(1,1,512,512)
    im2 = im2.view(1,1,512,512)


    out_p, desc = net(None, im1)
    kp = out_p['keypoints']
    kp = np.array(kp[0].cpu().numpy())


    p_1 = kp.astype(int)



    out_p, desc = net(None, im2)
    kp = out_p['keypoints']
    kp = np.array(kp[0].cpu().numpy())


    p_2 = kp.astype(int)




    batch_size=1
    p1 = torch.from_numpy(p_1).to(device).float().view(1,p_1.shape[0],2)
    p2 = torch.from_numpy(p_2).to(device).float().view(1,p_2.shape[0],2)

    out, desc = net(None, im1, my_p = p1 )
    out2, desc2 = net(None, im2, my_p = p2 )
    #desk1 = desc.transpose(2,3).contiguous().view(batch_size,256,-1)
    dsc_p1 = out['descriptors']
    pt_d1 = dsc_p1.transpose(1,2)
    d1 = pt_d1.to(device)
    t2 = time.time()

    dsc_p2 = out2['descriptors']
    pt_d2 = dsc_p2.transpose(1,2)
    d2 = pt_d2.to(device)

    ps1 = p1/256.
    ps2 = p2/256.
    t = time.time()
    ps1_input = ps1.repeat([1,1,2])
    ps2_input = ps2.repeat([1,1,2])
    t1 = time.time()

    out, point_map = f_net(None, d2, d1,  ps2_input, ps1_input, ocl= True)

    res = torch.argmax(point_map, dim=2)
    res = res.cpu().numpy()

    res = res[0]

    res[res==p_2.shape[0]] = -1
    scores = torch.max(point_map, dim=2)[0]
    scores = scores.detach().cpu().numpy()[0]
    res[scores<0.4] = -1

    my_p1 = p_1[res>=0]
    my_p2 = p_2[res[res>=0]]



M, mask = cv.findHomography(my_p1.astype(np.float32).reshape(-1,1,2), my_p2.astype(np.float32).reshape(-1,1,2), cv.RANSAC,6.0)

mask = mask.reshape(-1)

mask = mask==1
my_p1 = my_p1[mask]
my_p2 = my_p2[mask]

vis_or = np.concatenate((m1, m2), axis=1)
vis  = draw_lines(vis_or,my_p1.astype(int), my_p2.astype(int))


m1 =draw_interest_points(m1, p_1.astype(int))
m2 =    draw_interest_points(m2,p_2.astype(int) )

if(args.vis):

    cv2.imshow('vis',vis)
    cv2.waitKey(0)
if (args.save_image):
    cv2.imwrite('./results/res.jpg',vis )