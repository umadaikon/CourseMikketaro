#coding: utf-8
from Bmodules import resizer
import os
import shutil
import cv2
import sys

# name = "deadpoint2_1"
# video_file="../resource/deadpoint/deadpoint2_1/deadpoint2_1.mp4"
# image_dir="../resource/deadpoint2_1/img/origin/"+name+"/"
# image_file=name + "_" + "%s.png"
baseDir = sys.argv[1]
fileName = os.path.basename(sys.argv[1])
videoName = os.path.join(baseDir, fileName+".mp4")
imgDir = os.path.join(baseDir, "img")
imgName = fileName + "_" + "%s.png"

if not os.path.exists(imgDir):
    os.makedirs(imgDir)

i = 0
cap = cv2.VideoCapture(os.path.join(baseDir,videoName))
print("Saving frames...")
while(cap.isOpened()):
    flag, frame = cap.read()
    if flag == False:
      break
    frame = resizer.pixelResize(frame, 720, 1280)
    img = os.path.join(imgDir, imgName % str(i).zfill(4))
    cv2.imwrite(img, frame)
    i += 1
print("Save successful")
cap.release()
