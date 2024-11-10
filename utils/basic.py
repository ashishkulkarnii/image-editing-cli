from PIL import Image
import numpy as np
from matplotlib import colors
from datetime import datetime


def validate(val):
    if val > 255:
        return 255
    if val < 0:
        return 0
    return int(val)


def load(path):
    im1 = Image.open(r"{}".format(path))
    return im1


def bgr(colour):
    colour_rgb = Image.fromarray(
        np.array(([colors.hex2color(colors.cnames[colour])])))
    return [int(np.array(colour_rgb)[0][2] * 255), int(np.array(colour_rgb)[0][1] * 255), int(np.array(colour_rgb)[0][0] * 255)]


def save(im1):
    im1.save("{}.jpg".format(datetime.timestamp(datetime.now())))


def saveN(im1, name):
    im1.save("{}.jpg".format(name))


def saveNT(im1, name, type):
    im1.save("{}.{}".format(name, type))


def showImage(im1):
    im1.show()


color = {
    "red": [[155, 10, 0], [179, 255, 255]],
    "orange": [[5, 50, 50], [20, 255, 255]],
    "brown": [[5, 50, 10], [25, 255, 255]],
    "green": [[40, 40, 40], [70, 255, 255]],
    "blue": [[100, 150, 0], [140, 255, 255]],
}
