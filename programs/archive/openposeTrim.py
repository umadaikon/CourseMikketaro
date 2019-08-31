from Bmodules import trimmer
import cv2
import numpy as np
import os
import glob
import re

inDir = "../output/img/wally"
outDir = "../output/img/wally_trim"
if not os.path.exists(outDir):
    os.mkdir(outDir)

baseImg = cv2.imread("../input/img/wally_resize/wally000000_resize.png")
imgs = trimmer.trims(inDir, "*.png", baseImg.shape[0], baseImg.shape[1])

imgsname = glob.glob(os.path.join(inDir, "*.png"))
print("saving...")
for (name, im) in zip(imgsname, imgs):
    filename = re.split("\\\\|/", name)
    # print(filename)
    # print(outDir)
    # print(filename[-1])
    # print(glob.glob(os.path.join(outDir, filename[-1])))
    cv2.imwrite(os.path.join(outDir, filename[-1]), im)
print("save successful")
