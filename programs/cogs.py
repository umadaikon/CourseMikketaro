from Bmodules import cog
import numpy as np
import cv2
import glob
import os
import re
import csv
import datetime
import sys

baseDir = sys.argv[1]
fileName = os.path.basename(sys.argv[1])
renderedDir = os.path.join(baseDir, "rendered")
compjsonDir = os.path.join(baseDir, "compjson")
rejsonDir = os.path.join(baseDir, "rejson")

imgs = glob.glob(os.path.join(renderedDir, "*.png"))
if os.path.isdir(compjsonDir):
    jsons = glob.glob(os.path.join(compjsonDir, "*.json"))
else:
    jsons = glob.glob(os.path.join(rejsonDir, "*.json"))
outDir = "../output/"+fileName+"_{0:%Y%m%d%H%M}".format(datetime.datetime.now())
if not os.path.isdir(outDir):
    os.mkdir(outDir)
    os.mkdir(outDir+"/img")
coglist = list()
imglist = list()
print("CoGing...")
for (im, js) in zip(imgs, jsons):
    #重心の取得
    CoG = cog.openposeCoG(js)
    img = cv2.imread(im)
    if not CoG == -1:
        #重心座標をリストに保存
        coglist.append([CoG[0], CoG[1], CoG[6], CoG[7], CoG[8], CoG[9], CoG[10], CoG[11], CoG[2], CoG[3]])

        #重心の描画
        img = cv2.circle(img, (int(CoG[0]), int(CoG[1])), 10, (255,0,0), -1)
        #足元基準点の描画
        img = cv2.circle(img, (int(CoG[2]), int(CoG[3])), 10, (255,100,0), -1)
        #頭部基準点の描画
        img = cv2.circle(img, (int(CoG[4]), int(CoG[5])), 10, (255,100,100), -1)
    else:
        coglist.append([0, 0, 0, 0, 0, 0, 0, 0])
        #画像をリストに保存
    imglist.append(img)
    #画像の保存
    cv2.imwrite(os.path.join(outDir,"img",os.path.basename(im)), img)
print("CoG successful")
print("frames", len(coglist))

#重心座標のcsvファイルを出力
with open(os.path.join(outDir, fileName+'.csv'), 'w', newline='') as f:
    writer = csv.writer(f)  # writerオブジェクトを作成
    writer.writerow(['cogx', 'cogy', 'rightx', 'righty', 'leftx', 'lefty', 'neckx', 'necky'])
    writer.writerows(coglist)  # 内容を書き込む

videoShape = [imglist[0].shape[1], imglist[0].shape[0]]
#重心画像の動画を出力
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
video  = cv2.VideoWriter(os.path.join(outDir, fileName+'.mp4'), fourcc, 20.0, (videoShape[0], videoShape[1]))
for img in imglist:
    video.write(img)
video.release()
