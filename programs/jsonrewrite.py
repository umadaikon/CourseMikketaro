import numpy as np
import collections as cl
import re
import json
import os
import glob
import sys

baseDir = sys.argv[1]
jsonDir = os.path.join(baseDir, "json")
rejsonDir = os.path.join(baseDir, "rejson")
jsons = glob.glob(os.path.join(jsonDir, "*.json"))

print("Rewriting jsons...")
for i, js in enumerate(jsons):
    # print(i)
    filename = os.path.basename(js)
    f = open(js, "r")
    dict = json.load(f)

    p = cl.OrderedDict()
    p["No"] = int(re.search('\d+', filename).group()[-1])
    P = list()
    if len(dict['people']) < 2:
        #関節位置配列
        #対応位置はopenpose.pngを参照
        #[x座標,y座標,信頼度]
        keypoints = np.array(dict['people'][0]['pose_keypoints_2d']).reshape((18,3))
        for i, kpt in enumerate(keypoints):
            P.append([kpt[0],kpt[1]])
        #信頼度を追加
            P[i].append(kpt[2])
        # #値チェック
        # for i, p in enumerate(P):
        #     print(i,p)
    else:
        keypoints = list()
        sum = list()
        for people in dict['people']:
            keypoint = np.array(people['pose_keypoints_2d']).reshape((18,3))
            keypoints.append(keypoint)
            sum.append(np.sum(keypoint,0)[2])
        for i, kpt in enumerate(keypoints[np.argmax(sum)]):
            P.append([kpt[0],kpt[1]])
            P[i].append(kpt[2])
    p["keypoint"] = P
    with open(os.path.join(rejsonDir, filename), 'w') as f:
        json.dump(p, f, indent=4)
print("Rewrite successful")
