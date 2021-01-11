import numpy as np
import cv2
import matplotlib.pyplot as plt
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
parser.add_argument("--file_name", help="file_name",
                    default="frame_000940.png", type=str)

args = parser.parse_args()

save_path = args.save_path
depth_map_dir = args.depth_map_dir
ans_int_disp_map_dir = args.ans_int_disp_map_dir
file_name = args.file_name
min_depth = args.min_depth
max_depth = args.max_depth
bf = args.bf
d_inf = args.d_inf

# 可視化するファイル
file_names = []
file_names.append(file_name)
file_names_2 = []
file_names_2.append(file_name.split(".")[0])

# Check File Content

def draw_images(image_file):
    global save_path
    f_name = save_path + "/" + image_file
    gray_img = cv2.imread(f_name)
    return gray_img

for number, image in enumerate(file_names[0:1]):
    print(image)
    gray_img = draw_images(image)
    fig = plt.figure()
    plt.title("(a)", fontsize=22)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))
    fig.savefig("/home/ubuntu/data/Sayama/tmpsave/img_a.png")

# Depth Prediction Result
pred_depth = np.load(depth_map_dir + file_names_2[0] + '.npy')
pred_depth = cv2.resize(pred_depth, (416, 128))  # 次元数を2に変更

def draw_images_ans_int(image_file):
    global ans_int_disp_map_dir
    f_name = ans_int_disp_map_dir + "/" + image_file
    _ans_int_disp_map = cv2.imread(f_name)
    _ans_int_disp_map = cv2.cvtColor(_ans_int_disp_map, cv2.COLOR_RGB2GRAY)
    return _ans_int_disp_map

for number, image in enumerate(file_names[0:1]):
    print(image)
    ans_int_disp_map = draw_images_ans_int(image)

# 正解データの可視化
gt_depth = bf / (ans_int_disp_map - d_inf)
mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
gt_depth = gt_depth * mask
fig = plt.figure()
plt.imshow(gt_depth, cmap='magma')
plt.title("(b)", fontsize=22)
plt.axis('off')
plt.colorbar()
fig.savefig("/home/ubuntu/data/Sayama/tmpsave/img_b.png")

# 予測結果の可視化(スケール合わせ実施後)
scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
print(scalor)
pred_depth[mask] *= scalor
pred_depth[pred_depth < min_depth] = min_depth
pred_depth[pred_depth > max_depth] = max_depth
fig = plt.figure()
plt.imshow(pred_depth, cmap='magma')
plt.title("(c)", fontsize=22)
plt.axis('off')
plt.colorbar()
fig.savefig("/home/ubuntu/data/Sayama/tmpsave/img_c.png")
