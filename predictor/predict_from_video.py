from .src.identifier import Identifier
import cv2
import dlib
import sys
import time


def main():
    '''
    * @brief just main?
    '''
    identifier = Identifier(0.475)
    video_capture = cv2.VideoCapture(sys.argv[1])
    while True:
        ret, frame = video_capture.read()
        if ret is True:
            start = time.time()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            processed_frame = identifier.identify(
                small_frame)["processed_frame"]
            end = time.time()
            print(str(end - start) + "s")
            cv2.imshow('Video', processed_frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
