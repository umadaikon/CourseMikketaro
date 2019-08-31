import cv2
import numpy as np
import os
import glob
import math
import sys

def resize(baseImg, HEIGHT, WIDTH):
    rate = [baseImg.shape[0]/HEIGHT, baseImg.shape[1]/WIDTH]
    resize = [WIDTH, HEIGHT]

    # #数値確認
    # print('base',baseImg.shape)
    # print('resize', resize)
    def process(base):
        if base == 0:
            return int(np.round(baseImg.shape[1] * (1/rate[0]))), HEIGHT
        elif base == 1:
            return WIDTH, int(np.round(baseImg.shape[0] * (1/rate[1])))
        elif base == -1:
            return WIDTH, HEIGHT

    if baseImg.shape[0] > HEIGHT:
        if baseImg.shape[1] > WIDTH:
            if rate[0] >= rate[1]:
                resize = process(0)
            else:
                resize = process(1)
        else:
            resize = process(0)
    elif baseImg.shape[0] < HEIGHT:
        if baseImg.shape[1] >= WIDTH:
            resize = process(1)
        else:
            if rate[0] >= rate[1]:
                resize = process(0)
            else:
                resize = process(1)
    else: #baseImg.shape[0] == HEIGHT
        if baseImg.shape[1] > WIDTH:
            resize = process(1)
        elif baseImg.shape[1] < WIDTH:
            resize = process(0)
        else:
            resize = process(-1)
    # #数値確認
    # print(resize)
    return cv2.resize(baseImg, resize)

def resizes(input, imgName, HEIGHT, WIDTH):
#input ディレクトリパスか画像のリスト imgName ディレクトリパスの場合ファイルの正規表現
    images = list()
    if type(input) is str:
        input = glob.glob(os.path.join(input, imgName))
    print('resizing...')
    if type(input[0]) is str:
        for ms in input:
            images.append(resize(cv2.imread(ms), HEIGHT, WIDTH))
    else:
        for ms in input:
            images.append(resize(ms, HEIGHT, WIDTH))
    print('resize successful')
    return images

def pixelResize(baseImg, targetHEIGHT, targetWIDTH):
    baseHEIGHT = baseImg.shape[0]
    baseWIDTH = baseImg.shape[1]
    basePixel = baseHEIGHT * baseWIDTH
    targetPixel = targetHEIGHT * targetWIDTH
    gcd = math.gcd(baseHEIGHT, baseWIDTH)
    bH = baseHEIGHT / gcd
    bW = baseWIDTH / gcd
    mag = 1
    pixelDif = targetPixel - basePixel
    # print("Before:")
    # print("Base pixel", basePixel)
    # print("Target pixel", targetPixel)
    # print("Pixel dif", pixelDif)
    basePixel = 0
    while targetPixel > basePixel:
        pixelDif = targetPixel - basePixel
        mag = mag + 1
        baseHEIGHT = bH * mag
        baseWIDTH = bW * mag
        basePixel = baseHEIGHT * baseWIDTH
        # print("base shape", baseHEIGHT, baseWIDTH)
        # print("base pixel", basePixel)
        # print("pixelDif", targetPixel - basePixel)
    if pixelDif < basePixel-targetPixel:
        baseHEIGHT = bH * (mag-1)
        baseWIDTH = bW * (mag-1)
    basePixel = baseHEIGHT * baseWIDTH
    pixelDif = abs(targetPixel - basePixel)
    # print("After:")
    # print("Base pixel", basePixel)
    # print("Target pixel", targetPixel)
    # print("Pixel dif", pixelDif)
    resize = (int(baseWIDTH), int(baseHEIGHT))
    # print(resize)
    return cv2.resize(baseImg, resize)

# 動作チェック
if __name__ == '__main__':
    print('1.単体テスト 2.複数テスト 3.pixelResize')
    test = input('>> ')
    if test == '1':
        filename = 'wally000357.png'
        dst = resize(cv2.imread(filename), 720, 1280)
        print('dst',dst.shape)
        cv2.imshow('test',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        filesplit = filename.rsplit('.', 1)
        outputname = filesplit[0] + '_resize.' + filesplit[1]
        cv2.imwrite(outputname,dst)
    elif test == '2':
        img = resizes('./resource/', 'test*.png', 720, 1280)
        cv2.imshow("test", img[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif test == '3':
        filename = sys.argv[1]
        dst = pixelResize(cv2.imread(filename), 720, 1280)
        print('dst',dst.shape)
        cv2.imshow('test',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        filesplit = filename.rsplit('.', 1)
        outputname = filesplit[0] + '_Presize.' + filesplit[1]
        cv2.imwrite(outputname,dst)
    else:
        print('input error')
        input()
