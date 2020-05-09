import argparse
import numpy as np
import time
import cv2
from skimage.segmentation import slic, mark_boundaries
import train

def parse_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", required=True, help="Path to the image")
    args = parser.parse_args()
    return args


def pyramid(input_image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield input_image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(input_image.shape[1] / scale)
        input_image = cv2.resize(input_image, (w, w))
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if input_image.shape[0] < minSize[1] or input_image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield input_image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


def load_image(image_dir):
    # load the image
    input_image = cv2.imread(image_dir)

    # Using cv2.copyMakeBorder() method
    input_image = cv2.copyMakeBorder(input_image, 64, 64, 64, 64, cv2.BORDER_CONSTANT)

    return input_image


def slide_proc(input_image):
    # define the window width and height
    (winW, winH) = (128, 128)

    # loop over the image pyramid
    for resized in pyramid(input_image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.005)


def seg_N(input_image):
    h, w, dim = input_image.shape
    seg_number = max(h, w) * 0.1
    print('number of desired segments:   ', int(seg_number))
    return int(seg_number)


def super_pixel(input_image):
    super_ps = slic(input_image, n_segments=seg_N(input_image), compactness=15, sigma=1)
    for pixel_i in range(int(seg_N(input_image))):
        yield super_ps == pixel_i


def super_pproc(input_image):
    # loop over the image pyramid
    for resized in pyramid(input_image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for pixel_mask in super_pixel(resized):
            print(train.calc_haralick(pixel_mask))
            cv2.imshow('Window', mark_boundaries(resized, pixel_mask))
            cv2.waitKey(1)
            time.sleep(0.75)

if __name__ == '__main__':

    args = parse_args()
    input_image = load_image(args.image_dir)
    image_shape = input_image.shape
    print(image_shape)
    if image_shape[0] > 540 or image_shape[1] > 640:
        input_image = cv2.resize(input_image, (1000, 1000))
    # slide_proc(input_image)
    super_pproc(input_image)

