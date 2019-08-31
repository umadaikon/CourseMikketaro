import numpy as np
import math
import json
import cv2
import sys
import glob
import os

def getHeight(jsonDirPath):
    necks = list()
    bodys = list()
    rightFoots = list()
    rightLegs = list()
    leftFoots = list()
    leftLegs = list()
    #json読み取り
    jsonFilePaths = glob.glob(os.path.join(jsonDirPath, "*.json"))
    for i, jsonFilePath in enumerate(jsonFilePaths):
        # print(i)
        j = open(jsonFilePath, 'r')
        json_dict = json.load(j)

        #関節位置配列
        #対応位置はopenpose.pngを参照
        #[x座標,y座標,信頼度]
        P = list()

        #必要なデータを抜き取る
        for i, kpt in enumerate(json_dict['keypoint']):
            P.append([kpt[0],kpt[1]])
            #信頼度を追加
            P[i].append(kpt[2])
        # #値チェック
        # for i, p in enumerate(P):
        #     print(i,p)
        j.close()

        #1と(16,17)の中点の距離neck
        if not (P[1][2] == 0 or P[16][2] == 0 or P[17][2] == 0):
            necks.append(np.linalg.norm(np.array([P[1][0], P[1][1]])-np.array([(P[16][0]+P[17][0])/2, (P[16][1]+P[17][1])/2])))
        #1と(11,8)の中点の距離body
        if not (P[1][2] == 0 or P[11][2] == 0 or P[8][2] == 0):
            bodys.append(np.linalg.norm(np.array([(P[11][0]+P[8][0])/2, (P[11][1]+P[8][1])/2])-np.array([P[1][0], P[1][1]])))
        #8と9の距離rightFoot
        if not (P[8][2] == 0 or P[9][2] == 0):
            rightFoots.append(np.linalg.norm(np.array([P[8][0], P[8][1]])-np.array([P[9][0], P[9][1]])))
        #9と10の距離rightLeg
        if not (P[9][2] == 0 or P[10][2] == 0):
            rightLegs.append(np.linalg.norm(np.array([P[9][0], P[9][1]])-np.array([P[10][0], P[10][1]])))
        #11と12の距離leftFoot
        if not (P[11][2] == 0 or P[12][2] == 0):
            leftFoots.append(np.linalg.norm(np.array([P[11][0], P[11][1]])-np.array([P[12][0], P[12][1]])))
        #12と13の距離leftLeg
        if not (P[12][2] == 0 or P[13][2] == 0):
            leftLegs.append(np.linalg.norm(np.array([P[12][0], P[12][1]])-np.array([P[13][0], P[13][1]])))
    #neck+body+(長い方(rightFoot+rightLeg)or(leftFoot+leftLeg))
    neck = max(necks)
    body = max(bodys)
    rightFoot = max(rightFoots)
    rightLeg = max(rightLegs)
    leftFoot = max(leftFoots)
    leftLeg = max(leftLegs)

    height = neck + body + max(rightFoot+rightLeg, leftFoot+leftLeg)
    print("height", height)
    return height

if __name__ == '__main__':
    print(getHeight(sys.argv[1]))
