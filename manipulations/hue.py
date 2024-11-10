from utils.basic import validate, color

import numpy as np
from PIL import Image
from matplotlib import colors
import cv2


def greyscale(im):
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            im[i][j][0] = im[i][j][0] * 0.2898 + \
                im[i][j][1] * 0.5870 + im[i][j][2] * 0.1140
            im[i][j][1] = im[i][j][0]
            im[i][j][2] = im[i][j][0]
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im


def invert(im):
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            im[i][j][0] = 255 - im[i][j][0]
            im[i][j][1] = 255 - im[i][j][1]
            im[i][j][2] = 255 - im[i][j][2]
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im


def solarize(im, lt_gt, threshold):
    threshold = validate(threshold)
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            match lt_gt:
                case '<':
                    if im[i][j][0] < threshold:
                        im[i][j][0] = 255 - im[i][j][0]
                    if im[i][j][1] < threshold:
                        im[i][j][1] = 255 - im[i][j][1]
                    if im[i][j][2] < threshold:
                        im[i][j][2] = 255 - im[i][j][2]
                case '>':
                    if im[i][j][0] > threshold:
                        im[i][j][0] = 255 - im[i][j][0]
                    if im[i][j][1] > threshold:
                        im[i][j][1] = 255 - im[i][j][1]
                    if im[i][j][2] > threshold:
                        im[i][j][2] = 255 - im[i][j][2]
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im


def contrast(im, val):  # for val ∈ [-100, 100]
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            f = 103*(val + 99)/(99*(103-val))
            im[i][j][0] = validate(f * (im[i][j][0] - 128) + 128)  # r
            im[i][j][1] = validate(f * (im[i][j][1] - 128) + 128)  # g
            im[i][j][2] = validate(f * (im[i][j][2] - 128) + 128)  # b
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im


def brightness(im, val):  # for val ∈ [-100, 100]
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            im[i][j][0] = validate(im[i][j][0] + val*(255/100))  # r
            im[i][j][1] = validate(im[i][j][1] + val*(255/100))  # g
            im[i][j][2] = validate(im[i][j][2] + val*(255/100))  # b
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im


def gamma(im, val):
    gammaCorrection = 1 / val
    im = np.array(im)
    for i in range(len(im)):
        for j in range(len(im[0])):
            im[i][j][0] = 255 * (im[i][j][0]/255)**gammaCorrection  # r
            im[i][j][1] = 255 * (im[i][j][1]/255)**gammaCorrection  # g
            im[i][j][2] = 255 * (im[i][j][2]/255)**gammaCorrection  # b
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    im = Image.fromarray(im)
    print("100.000% converted.")
    return im

def cPop(im, colour):
    image = np.array(im)
    result = image[:, :, ::-1].copy()
    image = image[:, :, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array(color[colour][0])
    upper_white = np.array(color[colour][1])
    image_mark = image.copy()
    mask = cv2.inRange(image_mark, lower_white, upper_white)
    mask_inv = cv2.bitwise_not(mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = image.shape
    image = image[0:rows, 0:cols]
    colored_portion = cv2.bitwise_or(result, result, mask=mask)
    colored_portion = colored_portion[0:rows, 0:cols]
    gray_portion = cv2.bitwise_or(gray, gray, mask=mask_inv)
    gray_portion = np.stack((gray_portion,)*3, axis=-1)
    output = colored_portion + gray_portion
    mask = np.stack((mask,)*3, axis=-1)
    table_of_images = np.concatenate((result, mask, output), axis=1)
    cv2.imwrite("table.jpg", table_of_images)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output)


def cIPop(im, colour):
    image = np.array(im)
    result = image[:, :, ::-1].copy()
    image = image[:, :, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array(color[colour][0])
    upper_white = np.array(color[colour][1])
    image_mark = image.copy()
    mask = cv2.inRange(image_mark, lower_white, upper_white)
    mask_inv = cv2.bitwise_not(mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = image.shape
    image = image[0:rows, 0:cols]
    colored_portion = cv2.bitwise_or(result, result, mask=mask_inv)
    colored_portion = colored_portion[0:rows, 0:cols]
    gray_portion = cv2.bitwise_or(gray, gray, mask=mask)
    gray_portion = np.stack((gray_portion,)*3, axis=-1)
    output = colored_portion + gray_portion
    mask_inv = np.stack((mask_inv,)*3, axis=-1)
    table_of_images = np.concatenate((result, mask_inv, output), axis=1)
    cv2.imwrite("table.jpg", table_of_images)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output)
