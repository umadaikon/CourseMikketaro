import cv2
import numpy as np
import os
import glob

def accessfiles(path, regex):
    files = glob.glob(os.path.join(path,regex))
    return files

# 動作チェック
if __name__ == '__main__':
    png = accessfiles('./dir', '*.png')
    print(png)
