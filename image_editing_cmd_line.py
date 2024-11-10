from utils.basic import load, save, saveNT, saveN, showImage, bgr
from manipulations.hue import (
    greyscale,
    invert,
    solarize,
    contrast,
    brightness,
    gamma,
    cPop,
    cIPop,
)
from manipulations.blur import meanBlur, gaussBlur


def resize(im, x, y):
    return im.resize((x, y))


def currUpdate(ch):
    global curr, curr_max
    if ch == "+":
        curr += 1
        if curr > curr_max:
            curr_max = curr
    if ch == "-":
        curr -= 1


def firstLoad():
    global im, curr
    cmd = input(">>> load ")
    cmd = cmd.split(" ")
    im.append(load(cmd[0]))
    curr += 1


def getIn():
    global im, curr, undone, curr_max
    cmd = input(">>> ")
    cmd = cmd.split(" ")
    match cmd[0]:
        case "help":
            print(
                """Note:\t* [] means optional
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
* bgr <color name in English>"""
            )
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
            currUpdate("+")
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
            currUpdate("+")
        case "invert":
            im.append(invert(im[curr]))
            currUpdate("+")
        case "solarize":
            if len(cmd) >= 3:
                im.append(solarize(im[curr], cmd[1], float(cmd[2])))
                currUpdate("+")
        case "contrast":
            if len(cmd) >= 2:
                im.append(contrast(im[curr], float(cmd[1])))
                currUpdate("+")
        case "resize":
            if len(cmd) >= 3:
                im.append(resize(im[curr], int(cmd[1]), int(cmd[2])))
                currUpdate("+")
        case "brightness":
            if len(cmd) >= 2:
                im.append(brightness(im[curr], float(cmd[1])))
                currUpdate("+")
        case "gamma":
            if len(cmd) >= 3:
                if cmd[1] == "correction":
                    im.append(gamma(im[curr], float(cmd[2])))
                    currUpdate("+")
        case "color":
            if len(cmd) == 3:
                if cmd[1] == "pop":
                    im.append(cPop(im[curr], cmd[2]))
                    currUpdate("+")
            if len(cmd) == 4:
                if cmd[1] == "pop" and cmd[3] == "invert":
                    im.append(cIPop(im[curr], cmd[2]))
                    currUpdate("+")
        case "mean":
            if len(cmd) >= 3:
                if cmd[1] == "blur":
                    im.append(meanBlur(im[curr], int(cmd[2])))
                    currUpdate("+")
        case "gaussian":
            if len(cmd) == 3:
                if cmd[1] == "blur":
                    im.append(gaussBlur(im[curr], int(cmd[2]), 1))
                    currUpdate("+")
            elif len(cmd) >= 4:
                if cmd[1] == "blur":
                    im.append(gaussBlur(im[curr], int(cmd[2]), float(cmd[3])))
                    currUpdate("+")
        case "bgr":
            if len(cmd) >= 2:
                print(bgr(cmd[1]))
        case "undo":
            if curr != 0:
                undone.append(im.pop())
                currUpdate("-")
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
