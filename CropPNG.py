import os
import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", help="video_path", type=str)
parser.add_argument("--WIDTH", help="width of the output training image", default=416, type=int)
parser.add_argument("--HEIGHT", help="height of the output training image", default=128, type=int)
parser.add_argument("--INPUT_TXT_FILE", help="calib_cam_to_cam.txt path", default="./calib_cam_to_cam.txt", type=str)
parser.add_argument("--SEQ_LENGTH", help="result seq length", default=3, type=int)
parser.add_argument("--STEPSIZE", help="result step size", default=1, type=int)
parser.add_argument("--OUTPUT_DIR", help="result output dir", type=str)
parser.add_argument("--TEMP_DIR", help="temp data dir", type=str)

args = parser.parse_args()

base_path = args.base_path
WIDTH = args.WIDTH
HEIGHT = args.HEIGHT
INPUT_TXT_FILE = args.INPUT_TXT_FILE
SEQ_LENGTH = args.SEQ_LENGTH
STEPSIZE = args.STEPSIZE
OUTPUT_DIR = args.OUTPUT_DIR
TEMP_DIR = args.TEMP_DIR

data_dirs = [f.name for f in os.scandir(base_path) if not f.name.startswith('.')]


def make_dataset():
    # This function should be modified if you don't use KITTI dataset!
    global number_list, TEMP_DIR, WIDTH, HEIGHT
    if not TEMP_DIR.endswith('/'):
        TEMP_DIR = TEMP_DIR + '/'
    number_list = []
    for dataset in data_dirs:
        data_year = "2020"
        data_month = "08"
        data_date = "04"
        # Please designate these three variables if you don't use KITTI dataset
        # The value is not important
        IMAGE_DIR = base_path + dataset + "/"
        file_names = [f.name for f in os.scandir(IMAGE_DIR) if not f.name.startswith('.')]
        OUTPUT_DIR1 = TEMP_DIR + data_year + "_" + data_month + "_" + data_date + "/" + dataset + '/image_02/data'
        if not os.path.exists(OUTPUT_DIR1 + "/"):
            os.makedirs(OUTPUT_DIR1 + "/")
        make_dataset1(OUTPUT_DIR1, file_names, IMAGE_DIR, WIDTH, HEIGHT)


def make_dataset1(OUTPUT_DIR1, file_names, IMAGE_DIR, WIDTH, HEIGHT):
    for i in range(0, len(file_names)):
        image_file = IMAGE_DIR + file_names[i]
        img = cv2.imread(image_file)

        init_height, init_width = img.shape[:2]

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
        cv2.imwrite(OUTPUT_DIR1 + '/' + file_names[i], img)

make_dataset()
