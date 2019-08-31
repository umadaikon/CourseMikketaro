import numpy as np
import math
import json
import cv2

def openposeCoG(jsonFilePath):
    #傾き
    def inclination(x1,y1,x2,y2):
        return (y2-y1)/(x2-x1)
    #切片
    def intercept(d,x,y):
        return y-d*x
    #2直線の交点x座標
    def intersection(d1,b1,d2,b2):
        return (b2-b1)/(d1-d2)

    #重心座標
    cogs = list()
    #json読み取り
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
    #足元----------
    #足先2点の中点
    f0 = [(P[13][0]+P[10][0])/2, (P[13][1]+P[10][1])/2]

    #足先2点の直線f0
    #傾き
    d0 = inclination(P[10][0],P[10][1],P[13][0],P[13][1])
    #切片
    b0 = intercept(d0,P[13][0],P[13][1])

    #fに直交する傾き
    dd0 = -1*(1/d0)

    #膝2点の中点を通りfに直行する直線f1
    #切片
    b1 = intercept(dd0, (P[12][0]+P[9][0])/2, (P[12][1]+P[9][1])/2)
    #fとの交点f1
    f1 = [intersection(d0,b0,dd0,b1)]
    f1.append(dd0*f1[0]+b1)

    #腰2点の中点を通りfに直行する直線f2
    #切片
    b2 = intercept(dd0, (P[11][0]+P[8][0])/2, (P[11][1]+P[8][1])/2)
    #fとの交点f2
    f2 = [intersection(d0,b0,dd0,b2)]
    f2.append(dd0*f2[0]+b2)

    #f,f1,f2の重心点under
    under = [(f0[0]+f1[0]+f2[0])/3, (f0[1]+f1[1]+f2[1])/3]
    # if math.isnan(under[0]) or math.isnan(under[1]):
    #     continue;
    #足元---------

    #頭----------
    centerID = 0
    if not (P[16][0] == 0 or P[16][1] == 0 or P[17][0] == 0 or P[17][1] == 0):
        #右耳と左耳の中点
        center = [(P[16][0]+P[17][0])/2, (P[16][1]+P[17][1])/2]
        centerID = 1
    else:
        # 鼻と首の中点
        center = [(P[0][0]+P[1][0])/2, (P[0][1]+P[1][1])/2]
        centerID = 2
    #underからcenterへの直線f3
    #傾き
    d3 = inclination(center[0],center[1],under[0],under[1])
    #切片
    b3 = intercept(d3,center[0],center[1])

    #f3に直交する傾き
    dd3 = -1*(1/d3)

    #首を通りf3に直交する直線f4
    #切片
    b4 = intercept(dd3, P[1][0], P[1][1])
    #f3との交点
    f4 = [intersection(d3, b3, dd3, b4)]
    f4.append(dd3*f4[0]+b4)
    #ベクトルを生成
    f4v = np.array([f4[0], f4[1]])

    if centerID == 1:
        #f3とf4の交点とcenterの距離
        l = np.linalg.norm(center-f4v)
    else:
        #鼻を通りf3に直交する直線f5
        #切片
        b5 = intercept(dd3, P[0][0], P[0][1])
        #f3との交点
        f5 = [intersection(d3, b3, dd3, b5)]
        f5.append(dd3 * f5[0] + b5)
        #ベクトルを生成
        f5v = np.array([f5[0], f5[1]])

        #f3とf4の交点とf3とf5の交点の距離
        l = np.linalg.norm(f5v-f4v)

    #underからcenterへの単位ベクトルuを生成
    #underからcenterへのベクトルv
    v = np.array([center[0]-under[0],center[1]-under[1]])
    #正規化
    u = v / np.linalg.norm(v)

    #f3とf4の交点からベクトルuの向きにlを2倍伸ばした座標の点top
    top = [f4[0] + u[0]*l*2, f4[1] + u[1]*l*2]
    # if math.isnan(top[0]) or math.isnan(top[1]):
    #     continue;
    #頭----------

    #underからtopへのベクトル
    utvec = np.array([top[0]-under[0],top[1]-under[1]])
    #重心点
    cog__ = [under[0] + utvec[0]*0.6, under[1] + utvec[1]*0.6]

    cog_ = list()
    #重心の座標をappend
    cog_.append(cog__[0])
    cog_.append(cog__[1])
    #足元基準点の座標をappend
    cog_.append(under[0])
    cog_.append(under[1])
    #頭部基準点の座標をappend
    cog_.append(top[0])
    cog_.append(top[1])
    #右手の座標をappend
    cog_.append(P[4][0])
    cog_.append(P[4][1])
    #左手の座標をappend
    cog_.append(P[7][0])
    cog_.append(P[7][1])
    #首の座標をappend
    cog_.append(P[1][0])
    cog_.append(P[1][1])
    #信頼度の合計をappend
    cog_.append(np.sum(P,0)[2])
    cogs.append(cog_)


    # #要点確認
    # img = cv2.imread('./resource/wally000024_resize_rendered.png')
    # img = cv2.circle(img, (int(center[0]), int(center[1])), 10, (0,0,255), -1)
    # img = cv2.circle(img, (int(top[0]), int(top[1])), 10, (0,100,100), -1)
    # img = cv2.circle(img, (int(under[0]), int(under[1])), 10, (100,100,0), -1)
    # img = cv2.circle(img, (int(cog_[0]), int(cog_[1])), 10, (100,100,100), -1)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # #画像出力
    # print("save this image?")
    # save = input("0,1 or n >> ")
    # if save == '0' or save == '1':
    #     cv2.imwrite("./output/024_"+save+".png", img)

    # #値の確認
    # print("under:",under[0], under[1])
    # print("top:",top[0], top[1])
    # print("cog", cog[0], cog[1])
    # print(cogs)
    cog = cogs[np.argmax(cogs, 0)[2]]
    return cog

if __name__ == '__main__':
    print('view', openposeCoG('./resource/re18_keypoints.json'))
