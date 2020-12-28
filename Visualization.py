import numpy as np
import cv2
import matplotlib.pyplot as plt

save_path = '/home/ubuntu/data/Sayama/tmpdir/2020_08_04/video1top_png/image_02/data/'

depth_map_dir = "/home/ubuntu/data/Sayama_202008/result_video1top_273486/"

ans_int_disp_map_dir = "/home/ubuntu/Sayama/tmpdir/2020_08_04/video1middle_png/image_02/data"

file_names = ["frame_000940.png"]

file_names_2 = ["frame_000940"]

# Parameters for Abs Rel Error Calculation

min_depth = 5
max_depth = 80
bf = 109.65
d_inf = 2.67

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
    print(_ans_int_disp_map.size)
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
