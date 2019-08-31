import cv2
import numpy as np
import os
import glob

def trim(baseImg, HEIGHT, WIDTH):
    img = baseImg[0:HEIGHT, 0:WIDTH]
    return img

def trims(input, imgName, HEIGHT, WIDTH):
#input ディレクトリパスか画像のリスト imgName ディレクトリパスの場合ファイルの正規表現
    imgs = list()
    if type(input) is str:
        input = glob.glob(os.path.join(input, imgName))
        print("trimming...")
        if type(input[0]) is str:
            for ms in input:
                imgs.append(trim(cv2.imread(ms), HEIGHT, WIDTH))
        else:
            for ms in input:
                imgs.append(trim(ms), HEIGHT, WIDTH)
        print("trim successful")
    return imgs

#動作チェック
if __name__ == '__main__':
    if __name__ == '__main__':
        print('1.単体テスト 2.複数テスト')
        test = input('>> ')
        if test == '1':
            baseImg = cv2.imread("./resource/test_resize.png")
            img = trim(cv2.imread("./resource/test_resize_rendered.png"), baseImg.shape[0], baseImg.shape[1])
            cv2.imshow("base", cv2.imread('./resource/test_resize_rendered.png'))
            cv2.imshow("sizeBase", baseImg)
            cv2.imshow("trimmed", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif test == '2':
            baseImg = cv2.imread("./resource/test_resize.png")
            imgs = trims('./resource', "test_resize_rendered*.png", baseImg.shape[0], baseImg.shape[1])
            cv2.imshow("0",imgs[0])
            cv2.imshow("1",imgs[1])
            cv2.imshow("sizebase", baseImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('input error')
            input()
