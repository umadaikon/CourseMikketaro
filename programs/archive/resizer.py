#比率をそろえたまま 画素数を同じにする のが正しい

import cv2
import numpy as np
import os
import glob

def resize(baseImg, HEIGHT, WIDTH):
    rate = [baseImg.shape[0]/HEIGHT, baseImg.shape[1]/WIDTH]
    resize = [WIDTH, HEIGHT]

    #数値確認
    print('base',baseImg.shape)
    print('resize', HEIGHT, WIDTH)
    print('rate', rate)

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
                print("1")
            else:
                resize = process(1)
                print("2")
        else:
            resize = process(0)
            print("3")
    elif baseImg.shape[0] < HEIGHT:
        if baseImg.shape[1] >= WIDTH:
            resize = process(1)
            print("4")
        else:
            if rate[0] >= rate[1]:
                resize = process(0)
                print("5")
            else:
                resize = process(1)
                print("6")
    else: #baseImg.shape[0] == HEIGHT
        if baseImg.shape[1] > WIDTH:
            resize = process(1)
            print("7")
        elif baseImg.shape[1] < WIDTH:
            resize = process(0)
            print("8")
        else:
            resize = process(-1)
            print("9")
    # #数値確認
    # print(resize)
    return cv2.resize(baseImg, resize)

def resizes(input, imgName, HEIGHT, WIDTH):
#input ディレクトリパスか画像のリスト imgName ディレクトリパスの場合ファイルの正規表現
    images = list()
    if type(input) is str:
        input = glob.glob(os.path.join(input, imgName))
    print('resizing...')
    for ms in input:
        images.append(resize(cv2.imread(ms), 3120, 4160))
    print('resize completed')

    return images

# 動作チェック
if __name__ == '__main__':
    print('1.単体テスト 2.複数テスト')
    test = input('>> ')
    if test == '1':
        filename = './148.jpg'
        dst = resize(cv2.imread(filename), 3120, 4160)
        print('dst',dst.shape)
        cv2.imshow('test',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("./18.png", dst)
    elif test == '2':
        img = resizes('../resource/crossmove_2/img', '*.png', 3120, 4160)
        cv2.imshow("test", img[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i, im in enumerate(img):
            cv2.imwrite("../resource/crossmove_2/img/"+str(i)+".png", im)
    else:
        print('input error')
        input()
