import math
from PIL import Image
from datetime import datetime
import numpy as np
import cv2
from matplotlib import colors
import scipy.stats as st

color = {
    "red": [[155, 10, 0], [179, 255, 255]],
    "orange": [[5, 50, 50], [20, 255, 255]],
    "brown": [[5, 50, 10], [25, 255, 255]],
    "green": [[40, 40, 40], [70, 255, 255]],
    "blue": [[100, 150, 0], [140, 255, 255]],
}


def load(path):
    im1 = Image.open(r"{}".format(path))
    return im1


def validate(val):
    if val > 255:
        return 255
    if val < 0:
        return 0
    return int(val)


def saturation(im, s):
    im = im.convert("HSV")


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


def resize(im, x, y):
    return im.resize((x, y))


def bgr(colour):
    colour_rgb = Image.fromarray(
        np.array(([colors.hex2color(colors.cnames[colour])])))
    return [int(np.array(colour_rgb)[0][2] * 255), int(np.array(colour_rgb)[0][1] * 255), int(np.array(colour_rgb)[0][0] * 255)]


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


def meanBlur(im, ks):
    new = im
    im = np.array(im)
    rows, columns, three = im.shape
    if(rows < ks or columns < ks):
        print("kernel size too high.")
        return Image.fromarray(im)
    new = np.array(new)
    for i in range(rows-ks):
        for j in range(columns-ks):
            sum = [0, 0, 0]
            for x in range(ks):
                for y in range(ks):
                    if i+y < rows and j+x < columns:
                        sum[0] += im[i+y][j+x][0]
                        sum[1] += im[i+y][j+x][1]
                        sum[2] += im[i+y][j+x][2]
            new[i][j][0] = sum[0] / (ks*ks)
            new[i][j][1] = sum[1] / (ks*ks)
            new[i][j][2] = sum[2] / (ks*ks)
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    new = Image.fromarray(new)
    print("100.000% converted.")
    return new


def gaussKernelGenerator(kernlen=21, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def gaussBlur(im, kernel_size, sigma):
    ks = int(kernel_size / 2) * 2 + 1
    new = im
    im = np.array(im)
    rows, columns, three = im.shape
    if(rows < ks or columns < ks):
        print("kernel size too high.")
        return Image.fromarray(im)
    new = np.array(new)
    kernel = gaussKernelGenerator(ks, sigma)
    for i in range(rows-int(ks/2+1)):
        for j in range(columns-int(ks/2+1)):
            sum = [0, 0, 0]
            if(i+ks/2 < rows and j+ks/2 < columns):
                for x in range(ks):
                    for y in range(ks):
                        if(i+y < rows and j+x < columns):
                            sum[0] += im[i+y][j+x][0] * kernel[x][y]
                            sum[1] += im[i+y][j+x][1] * kernel[x][y]
                            sum[2] += im[i+y][j+x][2] * kernel[x][y]
                new[int(i+ks/2)][int(j+ks/2)][0] = validate(sum[0])
                new[int(i+ks/2)][int(j+ks/2)][1] = validate(sum[1])
                new[int(i+ks/2)][int(j+ks/2)][2] = validate(sum[2])
        if i % 10 == 0:
            print("{:.3f}% converted.\r".format(100 * (i) / len(im)), end='')
    new = Image.fromarray(new)
    print("100.000% converted.")
    return new


def save(im1):
    im1.save("{}.jpg".format(datetime.timestamp(datetime.now())))


def saveN(im1, name):
    im1.save("{}.jpg".format(name))


def saveNT(im1, name, type):
    im1.save("{}.{}".format(name, type))


def showImage(im1):
    im1.show()


def currUpdate(ch):
    global curr, curr_max
    if ch == '+':
        curr += 1
        if curr > curr_max:
            curr_max = curr
    if ch == '-':
        curr -= 1


def firstLoad():
    global im, curr
    cmd = input(">>> load ")
    cmd = cmd.split(' ')
    im.append(load(cmd[0]))
    curr += 1


def getIn():
    global im, curr, undone, curr_max
    cmd = input(">>> ")
    cmd = cmd.split(' ')
    match cmd[0]:
        case "help":
            print("""Note:\t* [] means optional
\t* all values are taken as float unless mentioned otherwise
General Commands:
* load <filename>.<extension>
* save [<filename>] [<extension>]
* exit [save]
* show [<filename>.<extension>]
* undo
* redo
Image Manipulation Commands:
* greyscale
* invert
* solarize <"<" or ">"> <threshold value from 0 to 255>
* contrast <value from -100 to 100>
* resize <new number (integer) of rows> <new number (integer) of columns>
* brightness <value from -100 to 100>
* gamma correction <gamma value>
* color pop <color name in English> [invert]
* mean blur <kernel size (integer)>
* gaussian blur <kernel size (integer)> [<sigma value, default sigma = 1>]
* bgr <color name in English>""")
        case "exit":
            if len(cmd) >= 2:
                if cmd[1] == "save":
                    save(im[curr])
                    exit()
                else:
                    exit()
            else:
                exit()
        case "load":
            im.append(load(cmd[1]))
            currUpdate('+')
        case "save":
            if len(cmd) == 3:
                saveNT(im[curr], cmd[1], cmd[2])
            elif len(cmd) == 2:
                saveN(im[curr], cmd[1])
            else:
                save(im[curr])
        case "show":
            if len(cmd) == 1:
                showImage(im[curr])
            elif len(cmd) == 2:
                im1 = load(cmd[1])
                showImage(im1)
        case "greyscale":
            im.append(greyscale(im[curr]))
            currUpdate('+')
        case "invert":
            im.append(invert(im[curr]))
            currUpdate('+')
        case "solarize":
            if len(cmd) >= 3:
                im.append(solarize(im[curr], cmd[1], float(cmd[2])))
                currUpdate('+')
        case "contrast":
            if len(cmd) >= 2:
                im.append(contrast(im[curr], float(cmd[1])))
                currUpdate('+')
        case "resize":
            if len(cmd) >= 3:
                im.append(resize(im[curr], int(cmd[1]), int(cmd[2])))
                currUpdate('+')
        case "brightness":
            if len(cmd) >= 2:
                im.append(brightness(im[curr], float(cmd[1])))
                currUpdate('+')
        case "gamma":
            if len(cmd) >= 3:
                if cmd[1] == "correction":
                    im.append(gamma(im[curr], float(cmd[2])))
                    currUpdate('+')
        case "color":
            if len(cmd) == 3:
                if cmd[1] == "pop":
                    im.append(cPop(im[curr], cmd[2]))
                    currUpdate('+')
            if len(cmd) == 4:
                if cmd[1] == "pop" and cmd[3] == "invert":
                    im.append(cIPop(im[curr], cmd[2]))
                    currUpdate('+')
        case "mean":
            if len(cmd) >= 3:
                if cmd[1] == "blur":
                    im.append(meanBlur(im[curr], int(cmd[2])))
                    currUpdate('+')
        case "gaussian":
            if len(cmd) == 3:
                if cmd[1] == "blur":
                    im.append(gaussBlur(im[curr], int(cmd[2]), 1))
                    currUpdate('+')
            elif len(cmd) >= 4:
                if cmd[1] == "blur":
                    im.append(gaussBlur(im[curr], int(cmd[2]), float(cmd[3])))
                    currUpdate('+')
        case "bgr":
            if len(cmd) >= 2:
                print(bgr(cmd[1]))
        case "undo":
            if curr != 0:
                undone.append(im.pop())
                currUpdate('-')
        case "redo":
            if len(undone) > 0:
                im.append(undone.pop())
                curr += 1
        case _:
            pass


im = []
curr = -1
curr_max = -1
undone = []

print("Python Image Editing Command-line: UE20MA251 Project\nSemester 4: APR 2022\nSYNTAX:\tload <image name>.<extension>\nEnter 'help' for more information.")

firstLoad()
while True:
    getIn()
