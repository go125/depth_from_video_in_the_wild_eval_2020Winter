import os
import sys
import skimage.io
import numpy as np
import cv2
import shutil
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", help="KITTI path", default="/home/ubuntu/data/raw_data_KITTI/", type=str)
parser.add_argument("--ROOT_DIR", help="Mask_RCNN path", default="../Mask_RCNN", type=str)
parser.add_argument("--HEIGHT", help="Input frame height.", default=128, type=int)
parser.add_argument("--WIDTH", help="Input frame width.", default=416, type=int)
parser.add_argument("--INPUT_TXT_FILE", help="calib_cam_to_cam.txt path", default="./calib_cam_to_cam.txt", type=str)
parser.add_argument("--SEQ_LENGTH", help="Number of frames in sequence.", default=3, type=int)
parser.add_argument("--STEPSIZE", help="result step size", default=1, type=int)
parser.add_argument("--OUTPUT_DIR", help="result output dir", default="/home/ubuntu/data/kitti_result_all_20201225",
                    type=str)
parser.add_argument("--TEMP_DIR", help="temp data dir", default="/home/ubuntu/data/train_data_example_all_20201225/",
                    type=str)
args = parser.parse_args()

base_path = args.base_path
ROOT_DIR = os.path.abspath(args.ROOT_DIR)
HEIGHT = args.HEIGHT
WIDTH = args.WIDTH
INPUT_TXT_FILE = args.INPUT_TXT_FILE
SEQ_LENGTH = args.SEQ_LENGTH
STEPSIZE = args.STEPSIZE
OUTPUT_DIR = args.OUTPUT_DIR
TEMP_DIR = args.TEMP_DIR

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

test_dirs = ["2011_09_26_drive_0117",
             "2011_09_28_drive_0002"]

data_dirs = ["2011_09_26_drive_0001",
             "2011_09_26_drive_0002"]

def make_dataset():
    global number_list, TEMP_DIR, WIDTH, HEIGHT
    number_list = []
    for dataset in data_dirs:
        if dataset in test_dirs:
            continue
        data_year = dataset.split("_")[0]
        data_month = dataset.split("_")[1]
        data_date = dataset.split("_")[2]

        data_num = "02"
        IMAGE_DIR = base_path + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync/image_" + data_num + "/data/"

        file_names = [f.name for f in os.scandir(IMAGE_DIR) if not f.name.startswith('.')]

        OUTPUT_DIR1 = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync_" + data_num + '/image_02/data'

        if not os.path.exists(OUTPUT_DIR1 + "/"):
            os.makedirs(OUTPUT_DIR1 + "/")

        make_dataset1(OUTPUT_DIR1, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num)

        OUTPUT_DIR2 = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync_" + data_num + '/image_03/data'

        if not os.path.exists(OUTPUT_DIR2 + "/"):
            os.makedirs(OUTPUT_DIR2 + "/")

        make_mask_images(OUTPUT_DIR2, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num)

        data_num = "03"
        IMAGE_DIR = base_path + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync/image_" + data_num + "/data/"

        file_names = [f.name for f in os.scandir(IMAGE_DIR) if not f.name.startswith('.')]

        OUTPUT_DIR1 = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync_" + data_num + '/image_02/data'

        if not os.path.exists(OUTPUT_DIR1 + "/"):
            os.makedirs(OUTPUT_DIR1 + "/")

        make_dataset1(OUTPUT_DIR1, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num)

        OUTPUT_DIR2 = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/" + dataset + "_sync_" + data_num + '/image_03/data'

        if not os.path.exists(OUTPUT_DIR2 + "/"):
            os.makedirs(OUTPUT_DIR2 + "/")

        make_mask_images(OUTPUT_DIR2, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num)

        OUTPUT_TXT_FILE = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/calib_cam_to_cam.txt"
        shutil.copyfile(INPUT_TXT_FILE, OUTPUT_TXT_FILE)

    run_all()
    TXT_RESULT_PATH = OUTPUT_DIR
    with open(TXT_RESULT_PATH + "/train.txt", mode='w') as f:
        f.write('\n'.join(number_list))


def make_dataset1(OUTPUT_DIR1, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num):
    for i in range(0, len(file_names)):
        image_file = IMAGE_DIR + file_names[i]
        img = cv2.imread(image_file)
        init_height, init_width = img.shape[:2]
        # cv2.resize(img, (WIDTH,HEIGHT))
        # 補間法を最近傍法に変更(ステレオカメラ使用時に視差データが壊れることを防ぐため)
        # リサイズ方法も切り抜きを経由するものに変更(アスペクト比が崩れることを防ぐため)
        if (init_height / init_width) > (HEIGHT / WIDTH):
            small_height = int(init_height * (WIDTH / init_width))
            img = cv2.resize(img, (WIDTH, small_height), interpolation=cv2.INTER_NEAREST)
            img = img[(small_height // 2 - HEIGHT // 2):(small_height // 2 + HEIGHT // 2), 0: WIDTH]
        else:
            small_width = int(init_width * (HEIGHT / init_height))
            img = cv2.resize(img, (small_width, HEIGHT), interpolation=cv2.INTER_NEAREST)
            img = img[0:HEIGHT, (small_width // 2 - WIDTH // 2):(small_width // 2 + WIDTH // 2)]
        if not os.path.exists(OUTPUT_DIR1):
            os.makedirs(OUTPUT_DIR1)
        cv2.imwrite(OUTPUT_DIR1 + '/' + data_num + "_" + file_names[i] + '.jpg', img)


def make_mask_images(OUTPUT_DIR2, file_names, IMAGE_DIR, WIDTH, HEIGHT, data_num):
    for i in range(0, len(file_names)):
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
        init_height, init_width = image.shape[:2]

        if (init_height / init_width) > (HEIGHT / WIDTH):
            small_height = int(init_height * (WIDTH / init_width))
            image = cv2.resize(image, (WIDTH, small_height), interpolation=cv2.INTER_NEAREST)
            image = image[(small_height // 2 - HEIGHT // 2):(small_height // 2 + HEIGHT // 2), 0: WIDTH]
        else:
            small_width = int(init_width * (HEIGHT / init_height))
            image = cv2.resize(image, (small_width, HEIGHT), interpolation=cv2.INTER_NEAREST)
            image = image[0:HEIGHT, (small_width // 2 - WIDTH // 2):(small_width // 2 + WIDTH // 2)]

        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        # Prepare black image
        mask_base = np.zeros((image.shape[0], image.shape[1], image.shape[2]), np.uint8)
        after_mask_img = image.copy()
        color = (10, 10, 10)  # white
        number_of_objects = len(r['masks'][0, 0])
        mask_img = mask_base

        for j in range(0, number_of_objects):

            mask = r['masks'][:, :, j]

            mask_img = visualize.apply_mask(mask_base, mask, color, alpha=1)

            if not os.path.exists(OUTPUT_DIR2):
                os.makedirs(OUTPUT_DIR2)
        cv2.imwrite(OUTPUT_DIR2 + '/' + data_num + "_" + file_names[i] + '.jpg', mask_img)


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0] == start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3, 4))[0:3, 0:3]
            break
    file.close()
    return ret


def run_all():
    global number_list, OUTPUT_DIR, TEMP_DIR
    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'

    for d in glob.glob(TEMP_DIR + '*/'):
        file_calibration = d + 'calib_cam_to_cam.txt'
        calib_raw = [get_line(file_calibration, 'P_rect_02'), get_line(file_calibration, 'P_rect_03')]

        for d2 in glob.glob(d + '*/'):
            DIR_NAME = d2.split('/')[-2]
            if not os.path.exists(OUTPUT_DIR + DIR_NAME + "/"):
                os.makedirs(OUTPUT_DIR + DIR_NAME + "/")
            print('Processing sequence', DIR_NAME)
            for subfolder in ['image_02/data']:
                ct = 1
                calib_camera = calib_raw[0] if subfolder == 'image_02/data' else calib_raw[1]
                folder = d2 + subfolder
                # files = glob.glob(folder + '/*.png')
                files = glob.glob(folder + '/*.jpg')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)
                for i in range(SEQ_LENGTH, len(files) + 1, STEPSIZE):
                    imgnum = str(ct).zfill(10)
                    big_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
                    wct = 0

                    for j in range(i - SEQ_LENGTH, i):  # Collect frames for this sample.
                        img = cv2.imread(files[j])
                        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                        zoom_x = WIDTH / ORIGINAL_WIDTH
                        zoom_y = HEIGHT / ORIGINAL_HEIGHT

                        # Adjust intrinsics.
                        calib_current = calib_camera.copy()
                        calib_current[0, 0] *= zoom_x
                        calib_current[0, 2] *= zoom_x
                        calib_current[1, 1] *= zoom_y
                        calib_current[1, 2] *= zoom_y

                        calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        big_img[:, wct * WIDTH:(wct + 1) * WIDTH] = img
                        wct += 1
                    cv2.imwrite(OUTPUT_DIR + DIR_NAME + '/' + imgnum + '.png', big_img)
                    f = open(OUTPUT_DIR + DIR_NAME + '/' + imgnum + '_cam.txt', 'w')
                    f.write(calib_representation)
                    f.close()
                    ct += 1

            for subfolder in ['image_03/data']:
                ct = 1
                calib_camera = calib_raw[0] if subfolder == 'image_02/data' else calib_raw[1]
                folder = d2 + subfolder
                # files = glob.glob(folder + '/*.png')
                files = glob.glob(folder + '/*.jpg')
                files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
                files = sorted(files)
                for i in range(SEQ_LENGTH, len(files) + 1, STEPSIZE):
                    imgnum = str(ct).zfill(10)
                    big_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
                    wct = 0

                    for j in range(i - SEQ_LENGTH, i):  # Collect frames for this sample.
                        img = cv2.imread(files[j])
                        ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape

                        zoom_x = WIDTH / ORIGINAL_WIDTH
                        zoom_y = HEIGHT / ORIGINAL_HEIGHT

                        # Adjust intrinsics.
                        calib_current = calib_camera.copy()
                        calib_current[0, 0] *= zoom_x
                        calib_current[0, 2] *= zoom_x
                        calib_current[1, 1] *= zoom_y
                        calib_current[1, 2] *= zoom_y

                        calib_representation = ','.join([str(c) for c in calib_current.flatten()])

                        img = cv2.resize(img, (WIDTH, HEIGHT))
                        big_img[:, wct * WIDTH:(wct + 1) * WIDTH] = img
                        wct += 1
                    # cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '.png', big_img)
                    # f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
                    cv2.imwrite(OUTPUT_DIR + DIR_NAME + '/' + imgnum + '-fseg.png', big_img)
                    f = open(OUTPUT_DIR + DIR_NAME + '/' + imgnum + '_cam.txt', 'w')
                    number_list.append(DIR_NAME + " " + imgnum)
                    f.write(calib_representation)
                    f.close()
                    ct += 1


make_dataset()
