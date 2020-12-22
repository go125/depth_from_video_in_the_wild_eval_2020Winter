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


def crop(img, segimg, fx, fy, cx, cy):
    # Perform center cropping, preserving 50% vertically.
    middle_perc = 0.50
    left = 1 - middle_perc
    half = left / 2
    a = img[int(img.shape[0] * (half)):int(img.shape[0] * (1 - half)), :]
    aseg = segimg[int(segimg.shape[0] * (half)):int(segimg.shape[0] * (1 - half)), :]
    cy /= (1 / middle_perc)

    # Resize to match target height while preserving aspect ratio.
    wdt = int((128 * a.shape[1] / a.shape[0]))
    x_scaling = float(wdt) / a.shape[1]
    y_scaling = 128.0 / a.shape[0]
    b = cv2.resize(a, (wdt, 128))
    bseg = cv2.resize(aseg, (wdt, 128))

    # Adjust intrinsics.
    fx *= x_scaling
    fy *= y_scaling
    cx *= x_scaling
    cy *= y_scaling

    # Perform center cropping horizontally.
    remain = b.shape[1] - 416
    cx /= (b.shape[1] / 416)
    c = b[:, int(remain / 2):b.shape[1] - int(remain / 2)]
    cseg = bseg[:, int(remain / 2):b.shape[1] - int(remain / 2)]

    return c, cseg, fx, fy, cx, cy


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
