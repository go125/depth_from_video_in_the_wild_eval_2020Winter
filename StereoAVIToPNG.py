import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_avi", help="video_path", type=str)
parser.add_argument("--path_output_png", help="output png path", type=str)
parser.add_argument("--option", help="top or middle or bottom", type=str)
parser.add_argument("--fps", help="output_fps", type=str)

args = parser.parse_args()

path_avi = args.path_avi
path_output_png = args.path_output_png
option = args.option
fps = float(args.fps)

def extract_frames(path_avi, path_output_png, option, fps):
    num = 0
    if not os.path.exists(path_output_png):
        os.makedirs(path_output_png)

    cap = cv2.VideoCapture(path_avi)
    count = 0
    avi_fps = 30
    freq = int(avi_fps / fps)

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            num += 1
            if num % freq == 0:
                init_height, init_width = frame.shape[:2]
                height = init_height // 3
                width = init_width
                if option == "top":
                    frame = frame[0:height, 0:width]
                if option == "middle":
                    frame = frame[height:(height * 2), 0:width]
                if option == "bottom":
                    frame = frame[(height * 2):(height * 3), 0:width]
                cv2.imwrite(os.path.join(path_output_png, "frame_{:06d}.png".format(count)), frame)
                count += 1
            if count % 100 == 0:
                print(count)
        else:
            break

    cap.release()

extract_frames(path_avi, path_output_png, option)
