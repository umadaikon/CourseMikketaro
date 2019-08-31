#python selectcsv.py 出力ディレクトリ絶対パス ムーブ前フレーム　ムーブ後フレーム 主体手（左0 右1）
import csv
import sys
import os
import glob
import cv2
from Bmodules import calcHeight

preMove = list()
postMove = list()
passMove = list()
baseDir = sys.argv[1]
mainhand = sys.argv[4]
csvPath = glob.glob(os.path.join(baseDir, "*.csv"))
fileName = os.path.splitext(os.path.basename(sys.argv[1]))[0].rsplit("_", 1)[0] + ".csv"
# print(os.path.join(baseDir, "img", "*"+sys.argv[2]+"_rendered.png"))
# standardHeight = 494.3399048805102
standardHeight = 639.4899329266506
targetHeight = calcHeight.getHeight(os.path.join(baseDir, "height"))
check = cv2.imread(os.path.join(baseDir, "img", "*"+sys.argv[2]+".png"))
with open(csvPath[0], 'r') as fr:
    list = list(csv.reader(fr))
    del list[0]
    # print(len(list))
    preMove = list[int(sys.argv[2])]
    postMove = list[int(sys.argv[3])]


rightx_b = float(preMove[2])
righty_b = float(preMove[3])
leftx_b = float(preMove[4])
lefty_b = float(preMove[5])

cogx_a = float(postMove[0])
cogy_a = float(postMove[1])
rightx_a = float(postMove[2])
righty_a = float(postMove[3])
leftx_a = float(postMove[4])
lefty_a = float(postMove[5])
neckx_a = float(postMove[6])
necky_a = float(postMove[7])

if mainhand == "0": #軸手が右
    #x座標の処理
    dx = rightx_b - rightx_a
    cogx_a = cogx_a + dx
    rightx_a = rightx_a + dx
    leftx_a = leftx_a + dx
    neckx_a = neckx_a + dx
    #y座標の処理
    dy = righty_b - righty_a
    cogy_a = cogy_a + dy
    righty_a = righty_a + dy
    lefty_a = lefty_a + dy
    necky_a = necky_a + dy
elif mainhand == "1": #軸手が左
    #x座標の処理
    dx = leftx_b - leftx_a
    cogx_a = cogx_a + dx
    rightx_a = rightx_a + dx
    leftx_a = leftx_a + dx
    neckx_a = neckx_a + dx
    #y座標の処理
    dy = lefty_b - lefty_a
    cogy_a = cogy_a + dy
    righty_a = righty_a + dy
    lefty_a = lefty_a + dy
    necky_a = necky_a + dy
if not mainhand == "2":
    postMove = [cogx_a, cogy_a, rightx_a, righty_a, leftx_a, lefty_a, neckx_a, necky_a]

def calcPostmove(index, xy):
    return float(postMove[index])-float(preMove[xy])
if mainhand == '0' or mainhand == '2':
    print("moveLeft")
    passMove = [calcPostmove(0, 6), calcPostmove(1, 7), float(preMove[2])-float(preMove[4]), float(preMove[3])-float(preMove[5]), float(postMove[4])-float(preMove[4]), float(postMove[5])-float(preMove[5]), mainhand, sys.argv[2], sys.argv[3]]
    # check = cv2.arrowedLine(check, (int(float(preMove[2])), int(float(preMove[3]))), (int(float(preMove[4])), int(float(preMove[5]))), (0, 255, 0), thickness=4, tipLength=0.3)

elif mainhand == '1':
    print("moveRight")
    passMove = [calcPostmove(0, 6), calcPostmove(1, 7), float(preMove[4])-float(preMove[2]), float(preMove[5])-float(preMove[3]), float(postMove[2])-float(preMove[2]), float(postMove[3])-float(preMove[3]), mainhand, sys.argv[2], sys.argv[3]]
    # check = cv2.arrowedLine(check, (int(float(preMove[4])), int(float(preMove[5]))), (int(float(preMove[2])), int(float(preMove[3]))), (0, 255, 0), thickness=4, tipLength=0.3)

print(passMove)
passMove[0] = passMove[0] * (standardHeight/targetHeight)
passMove[1] = passMove[1] * (standardHeight/targetHeight)
passMove[2] = passMove[2] * (standardHeight/targetHeight)
passMove[3] = passMove[3] * (standardHeight/targetHeight)
passMove[4] = passMove[4] * (standardHeight/targetHeight)
passMove[5] = passMove[5] * (standardHeight/targetHeight)
print(passMove)

# cv2.imshow("pVec", check)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

passDir = os.path.join(baseDir, "pass")
if not os.path.isdir(passDir):
    os.mkdir(passDir)

with open(os.path.join(passDir, fileName), 'w', newline='') as fw:
    writer = csv.writer(fw)
    writer.writerow(['cogx', 'cogy', 'p_beforex', 'p_beforey', 'before_afterx', 'before_aftery', 'mainhand', 'preframe', 'postframe'])
    writer.writerow(passMove)
