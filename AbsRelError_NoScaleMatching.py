#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from xml.etree import ElementTree
import cv2
from PIL import Image
import tensorflow as tf
import csv
import argparse
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pickle
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", help="Raw images dir",
                    default='/home/ubuntu/data/Sayama/tmpdir/2020_08_04/video1top_png/image_02/data/', type=str)
parser.add_argument("--depth_map_dir", help="Depth maps dir",
                    default='/home/ubuntu/Sayama/result_video1top_273486/', type=str)
parser.add_argument("--ans_int_disp_map_dir", help="Disparity map",
                    default="/home/ubuntu/data/Sayama/tmpdir/2020_08_04/video1middle_png/image_02/data", type=str)
parser.add_argument("--min_depth", help="Abs Rel Error Calculation Settings.", default=5, type=int)
parser.add_argument("--max_depth", help="Abs Rel Error Calculation Settings.", default=80, type=int)
parser.add_argument("--bf", help="Stereo Camera Parameters.", default=109.65, type=float)
parser.add_argument("--d_inf", help="Stereo Camera Parameters.", default=2.67, type=float)

args = parser.parse_args()

save_path = args.save_path
depth_map_dir = args.depth_map_dir
ans_int_disp_map_dir = args.ans_int_disp_map_dir
min_depth = args.min_depth
max_depth = args.max_depth
bf = args.bf
d_inf = args.d_inf

# Making file list
# file_names = ["frame_000250"]
file_names = []
for file in os.listdir(save_path):
    if os.path.isfile(os.path.join(save_path, file)):
        file_name = file.rstrip('.png\n')
        file_names.append(file_name)

num_test = len(file_names)

rms = np.zeros(num_test, np.float32)
log_rms = np.zeros(num_test, np.float32)
abs_rel = np.zeros(num_test, np.float32)
sq_rel = np.zeros(num_test, np.float32)
d1_all = np.zeros(num_test, np.float32)
a1 = np.zeros(num_test, np.float32)
a2 = np.zeros(num_test, np.float32)
a3 = np.zeros(num_test, np.float32)
scalors = np.zeros(num_test, np.float32)


def draw_images_ans_int(image_file):
    global ans_int_disp_map_dir
    f_name = ans_int_disp_map_dir + "/" + image_file + ".png"
    ans_int_disp_map = cv2.imread(f_name)
    ans_int_disp_map = cv2.cvtColor(ans_int_disp_map, cv2.COLOR_RGB2GRAY)
    return ans_int_disp_map


def calc_center(xmin=0, ymin=0, img_height=128, img_width=416, clip_height=128, clip_width=416, dfv_height=128,
                dfv_width=416):
    center_ratio_x = (img_height // 2 - xmin) / clip_height
    center_ratio_y = (img_width // 2 - ymin) / clip_width
    center_x = int(dfv_height * center_ratio_x)
    center_y = int(dfv_width * center_ratio_y)
    return [center_x, center_y]


def abs_rel_error_single_image(i):
    pred_depth = np.load(depth_map_dir + file_names[i] + '.npy')
    init_height, init_width = pred_depth.shape[:2]
    pred_depth = cv2.resize(pred_depth, (init_width, init_height))

    ans_int_disp_map = draw_images_ans_int(file_names[i])

    gt_depth = bf / (ans_int_disp_map - d_inf)

    mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

    height = 1.2
    acceleration = 0.065
    theta = math.asin(acceleration)
    truth_z = height / math.sin(theta)
    center = calc_center()
    prezent_z = pred_depth[center[0]][center[1]]
    scalor = truth_z / prezent_z
    # scalor=1
    # scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
    scalors[i] = scalor

    pred_depth[mask] *= scalor

    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth

    abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print(str(i) + "/" + str(num_test))


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


for i in range(0, num_test):
    abs_rel_error_single_image(i)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} ".format('abs_rel', 'sq_rel', 'rms',
                                                                                       'log_rms', 'd1_all', 'a1', 'a2',
                                                                                       'a3', 'scalor'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} ,{:10.4f} ".format(abs_rel.mean(),
                                                                                                         sq_rel.mean(),
                                                                                                         rms.mean(),
                                                                                                         log_rms.mean(),
                                                                                                         d1_all.mean(),
                                                                                                         a1.mean(),
                                                                                                         a2.mean(),
                                                                                                         a3.mean(),
                                                                                                         scalors.mean()))
