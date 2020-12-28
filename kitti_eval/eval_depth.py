from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--test_file_list", type=str, default='./kitti_eval/test_files_eigen.txt',
                    help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()


def resize_with_black(img, base_w, base_h ,init_height, init_width):
    # https://github.com/Penguin8885/img_resizer
    base_ratio = base_w / base_h  # リサイズ画像サイズ縦横比
    img_h, img_w = init_height, init_width  # 画像サイズ
    img_ratio = img_w / img_h  # 画像サイズ縦横比

    black_img = np.zeros((base_h, base_w, 1), np.float32)  # ベース作成

    if img_ratio > base_ratio:
        h = int(base_w / img_ratio)  # 横から縦を計算
        w = base_w  # 横を合わせる
        resize_img = cv2.resize(img, (w, h))  # リサイズ
    else:
        h = base_h  # 縦を合わせる
        w = int(base_h * img_ratio)  # 縦から横を計算
        resize_img = cv2.resize(img, (w, h))  # リサイズ

    black_img[int(base_h / 2 - h / 2):int(base_h / 2 + h / 2),
    int(base_w / 2 - w / 2):int(base_w / 2 + w / 2)] = resize_img
    # オーバーレイ
    resize_img = black_img  # 上書き

    return resize_img


def make_aspect_mask(base_w, base_h ,init_height, init_width):
    # https://github.com/Penguin8885/img_resizer
    img = np.ones((init_height, init_width, 1), np.float32)
    base_ratio = base_w / base_h  # リサイズ画像サイズ縦横比
    img_h, img_w = init_height, init_width  # 画像サイズ
    img_ratio = img_w / img_h  # 画像サイズ縦横比

    black_img = np.zeros((base_h, base_w, 1), np.float32)  # ベース作成

    if img_ratio > base_ratio:
        h = int(base_w / img_ratio)  # 横から縦を計算
        w = base_w  # 横を合わせる
        resize_img = cv2.resize(img, (w, h))  # リサイズ
    else:
        h = base_h  # 縦を合わせる
        w = int(base_h * img_ratio)  # 縦から横を計算
        resize_img = cv2.resize(img, (w, h))  # リサイズ

    black_img[int(base_h / 2 - h / 2):int(base_h / 2 + h / 2),
    int(base_w / 2 - w / 2):int(base_w / 2 + w / 2)] = resize_img
    # オーバーレイ
    resize_img = black_img  # 上書き

    return resize_img

def main():
    global init_height, init_width
    # 初期値設定
    init_height, init_width = 128, 416

    # 予測値読み込み
    pred_depths = np.load(args.pred_file)

    # 予測データのサイズ合わせ
    test_files = read_text_lines(args.test_file_list)
    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(test_files, args.kitti_dir)
    num_test = len(im_files)
    gt_depths = []
    pred_depths_resized = []
    for t_id in range(num_test):
        print(t_id)
        camera_id = cams[t_id]  # 2 is left, 3 is right
        temp_pred_depth = pred_depths[t_id]
        init_height, init_width = temp_pred_depth.shape[:2]
        print("init_height=" + str(init_height))
        print("init_width=" + str(init_width))
        # 予測値のアスペクト比合わせ
        temp_pred_depth = resize_with_black(temp_pred_depth, 416, 128, init_height, init_width)
        # 予測値のサイズ合わせ
        pred_depths_resized.append(
            cv2.resize(temp_pred_depth,
                       (im_sizes[t_id][1], im_sizes[t_id][0]),
                       interpolation=cv2.INTER_LINEAR))
        depth = generate_depth_map(gt_calib[t_id],
                                   gt_files[t_id],
                                   im_sizes[t_id],
                                   camera_id,
                                   False,
                                   True)
        gt_depths.append(depth.astype(np.float32))
    pred_depths = pred_depths_resized

    # 評価値計算
    rms = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel = np.zeros(num_test, np.float32)
    d1_all = np.zeros(num_test, np.float32)
    a1 = np.zeros(num_test, np.float32)
    a2 = np.zeros(num_test, np.float32)
    a3 = np.zeros(num_test, np.float32)
    scalors = np.zeros(num_test, np.float32)

    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])
        mask = np.logical_and(gt_depth > args.min_depth,
                              gt_depth < args.max_depth)

        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # アスペクト比修正用マスクの合成
        mask_w, mask_h = mask.shape[:2]
        aspect_mask = make_aspect_mask(mask_w, mask_h, init_height, init_width)
        mask = np.logical_and(mask, aspect_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        scalors[i] = scalor
        pred_depth[mask] *= scalor

        # 個々の画像の指標値を計算
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} ".format('abs_rel', 'sq_rel', 'rms',
                                                                                           'log_rms', 'd1_all', 'a1',
                                                                                           'a2', 'a3', 'scalor'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f} ,{:10.4f} ".format(
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean(),
        scalors.mean()))


main()
