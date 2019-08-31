import json
import os
import glob
import collections as cl
import sys

baseDir = sys.argv[1]
rejsonDir = os.path.join(baseDir, "rejson")
compjsonDir = os.path.join(baseDir, "compjson")
jsons = glob.glob(os.path.join(rejsonDir, "*.json"))

P = list()
filename = list()

print("Searching lack json...")
for js in jsons:
    filename.append(os.path.basename(js))
    f = open(js, "r")
    dict = json.load(f)
    P.append(dict["keypoint"])

for frameNo, p in enumerate(P):
    # print(frameNo)
    for kptNo, kpt in enumerate(p):
        # print("frameNo", frameNo, "kptNo", kptNo, "kpt", kpt)
        if kpt[2] == 0:
            # print("Compliting")
            # print("frameNo", frameNo, "kptNo", kptNo, "kpt", kpt)
            times = 0
            found = 0
            i = frameNo
            prepoint = list()
            postpoint = list()
            #フレームが0番目でないとき前フレームを事前に取得
            if not frameNo == 0:
                prepoint = P[frameNo-1][kptNo]
                times += 1
                found += 1
                # print("pre", prepoint)
            #信頼度が0でない後フレームを見つけるまで繰り返し
            while found < 2 and i < len(filename)-1:
                times += 1
                i += 1
                # print("whiling...", i, P[i][kptNo][2])
                if not P[i][kptNo][2] == 0:
                    if found <= 0:
                        prepoint = P[i][kptNo]
                        found += 1
                        # print("pre", prepoint)
                    else:
                        postpoint = P[i][kptNo]
                        found += 1
                        # print("post", postpoint)
            if not found == 2:
                times = 0
                found = 0
                i = frameNo
                prepoint = list()
                postpoint = list()
                while found < 2 and i > 0:
                    times += 1
                    i -= 1
                    # print("backwhiling...", i, P[i][kptNo][2])
                    if not P[i][kptNo][2] == 0:
                        if found <= 0:
                            postpoint = P[i][kptNo]
                            found += 1
                            # print("post", postpoint)
                        else:
                            prepoint = P[i][kptNo]
                            found += 1
                            # print("pre", prepoint)

            #差分をtimesで割り、前フレームの値に足す
            if prepoint:
                P[frameNo][kptNo] = [prepoint[0]+(postpoint[0]-prepoint[0])/times,prepoint[1]+(postpoint[1]-prepoint[1])/times, -1]
                # print("Completed", P[frameNo][kptNo])
            else:
                P[frameNo][kptNo] = [0, 0, -1]

    p = cl.OrderedDict()
    p["No"] = frameNo
    p["keypoint"] = P[frameNo]
    with open(os.path.join(compjsonDir, filename[frameNo]), 'w') as f:
        json.dump(p, f, indent=4)
print("Complete successful")
