import cv2
import os


def extract_frames(path_avi, path_output_png, option="None"):
    num = 0
    if not os.path.exists(path_output_png):
        os.mkdir(path_output_png)

    cap = cv2.VideoCapture(path_avi)
    count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            num += 1
            if num % 3 == 0:
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
            if num % 300 == 0:
                print(count)
        else:
            break

    cap.release()
