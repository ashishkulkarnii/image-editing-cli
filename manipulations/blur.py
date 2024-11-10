from utils.basic import validate

import numpy as np
from PIL import Image
import scipy.stats as st

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
