import numpy as np
import csv
import math
import cv2
from sympy.geometry import Point, Polygon
import pylab as plt
import mysql.connector
import operator
import copy
import sys
import joblib
import os
from Bmodules import resizer

# outputShape = (831, 1108)
outputShape = (720, 1280)
resizeShape = (1500, 2700)
#基準身長
standardHeight = 165
# standardPixelHeight = 494.3399048805102
standardPixelHeight = 639.4899329266506

#傾き
def inclination(x1,y1,x2,y2):
	return (y2-y1)/(x2-x1)
#切片
def intercept(d,x,y):
	return y-d*x
#2直線の交点座標
def intersection(d1,b1,d2,b2):
	x = (b2-b1)/(d1-d2)
	y = d1 * x +b1
	return [x, y]
#直線の要素
def line(x1, y1, x2, y2):
	#傾き
	d = (y2-y1)/(x2-x1)
	#切片
	b = y1-d*x1
	return d, b

#彩度を下げる
def HLowSV(imgpass):
	hsv = cv2.cvtColor(resizer.pixelResize(cv2.imread(imgpass), outputShape[0], outputShape[1]), cv2.COLOR_BGR2HSV_FULL)
	hsv[:,:,1] = hsv[:,:,1] * 0.5
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

#座標と(0.0, 1.0)角度でソート
def BaseAngleSort(pointList):
	baseLine = np.array([0.0, 1.0])
	#角度（絶対値）算出
	angleList = [np.rad2deg(np.arccos(np.clip(np.inner(pl, np.array([0.0, 1.0]))/(np.linalg.norm(pl) * np.linalg.norm(np.array([0.0, 1.0]))), -1.0, 1.0))) for pl in pointList]
	#x軸が正のものの角度を90度を始めとした角度になるよう計算
	angleList = [360-al if pl[0] > 0 else al for al, pl in zip(angleList, pointList)]

	return [pl for al, pl in sorted(zip(angleList, pointList))]

	#90度（真下）を始めとしてソート
	# angle_ = list()
	# angle__ = list()
	# for pl, al in zip(pointList, angleList):
	# 	if al < 90.0:
	# 		angle_.append([pl, al])
	# 	else:
	# 		angle__.append([pl, al])
	# # nMRRA__ = [nmrra for nmrra in nMRRA if nmrra[1]>=90.0]
	# # nMRRA_ = [nmrra for nmrra in nMRRA if nmrra[1]<90.0]
	# angle_.sort(key=operator.itemgetter(1))
	# angle__.sort(key=operator.itemgetter(1))

	# return [pl for pl, al in angle_] + [pl for pl, al in angle__]

#描画関連変数
lineSize = 3
circle1Size = 10
circle2Size = 15
circle3Size = 20
fontSize = 1
fontThick = 2
lengFontPoint = circle1Size * 4
leftFontPoint = -circle1Size * 4
rightFontPoint = circle1Size
B = 255
point = -20

#ホールド中心
holdPoint = list()
#ホールド4点
holdRect = list()
#練習したいムーブ名
practiceMove = str()
#ムーヴ可能ホールドリスト
moveHoldList = list()
#移動可能ホールドリスト
nonMoveHoldList = list()
#始点
start = list()
#手
hand = int()
#身長
pHeight = float()

# wallImgPath = "./resource/IMG_9519.JPG"
wallImgPath = sys.argv[1]
# wallImg = resizer.pixelResize(cv2.imread(wallImgPath), outputShape[0], outputShape[1])
# wallImg = cv2.resize(wallImg, (outputShape[0], outputShape[1]))
wallImg = HLowSV(wallImgPath)
wallShape = [wallImg.shape[1], wallImg.shape[0]]

csvResizeX = wallShape[0]/resizeShape[0]
csvResizeY = wallShape[1]/resizeShape[1]

# with open("./resource/pi_9519.csv", 'r') as fr:
with open(sys.argv[2], 'r') as fr:
	reader = csv.reader(fr)
	header = next(reader)
	for rd in reader:
		holdPoint.append([float(rd[0])*csvResizeX, float(rd[1])*csvResizeY])
		holdRect.append([float(rd[2])*csvResizeX, float(rd[3])*csvResizeY, float(rd[4])*csvResizeX, float(rd[5])*csvResizeY, float(rd[6])*csvResizeX, float(rd[7])*csvResizeY, float(rd[8])*csvResizeX, float(rd[9])*csvResizeY])
# #値確認
# print("Holdpoint", holdPoint)
# print("Holdrect", holdRect)
#描画
# for i, hp in enumerate(holdPoint):
# 	wallImg = cv2.circle(wallImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 40, 255), -1)
for No, hr in enumerate(holdRect):
	wallImg = cv2.rectangle(wallImg, (int(hr[2]), int(hr[3])), (int(hr[6]), int(hr[7])), (0, 0, 255), lineSize)
for No, hr in enumerate(holdRect):
	wallImg = cv2.putText(wallImg, str(No), (int(hr[4]), int(hr[5])+35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 153, 252), 3)
cv2.imwrite("HoldNo.png", wallImg)

#事前入力
#身長
print("Input height")
pHeight = float((input(" > ")))
#ムーヴ
print("Select move")
print("deadpoint:1 crossmove:2 diagonal:3 dropknee:4 dyno:5")
practiceMove = int(input(" > "))
if practiceMove == 1:
	print("Select deadpoint")
	practiceMove = "deadpoint"
elif practiceMove == 2:
	print("Select crossmove")
	practiceMove = "crossmove"
elif practiceMove == 3:
	print("Select diagonal")
	practiceMove = "diagonal"
elif practiceMove == 4:
	print("Select dropknee")
	practiceMove = "dropknee"
elif practiceMove == 5:
	print("Select dyno")
	practiceMove = "dyno"
else:
	print("Input error", practiceMove)
	sys.exit()
#始点
print("Select start hold")
cv2.imshow("Select hold", cv2.resize(wallImg, (int(wallImg.shape[1]/2), int(wallImg.shape[0]/2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
StartHoldNo = input("holdNo > ")
start = [holdPoint[int(StartHoldNo)][0], holdPoint[int(StartHoldNo)][1]]
#手
print("Which hand?")
hand = int(input("hand > "))

#ムーヴ情報
moves = ["deadpoint", "crossmove", "diagonal", "dropknee", "dyno"]
#0を原点としたmove情報
moveRawRange = list()
#moveRawRange = [[ムーヴ名, 原点に近いp1, 原点に近いp2]]
#0を原点としたnonmove情報
nonMoveRawRange = list()
MoveHoldsRange = list()

#----------全ムーヴデータ取得
moveName = str()
# count = 1
for mvs in moves:
	moveName = mvs
	# if count == 1:
	# 	moveName = "deadpoint"
	# elif count == 2:
	# 	moveName = "crossmove"
	# elif count == 3:
	# 	moveName = "diagonal"
	# elif count == 4:
	# 	moveName = "dropknee"
	# elif count == 5:
	# 	moveName = "dyno"
	# print("Count", count)
	conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='test3_db'
	)

	cur = conn.cursor()

	# # value = "values(" + str(id) + ", " + str(p_beforex) + ", " + str(p_beforey) + ", " + str(b_ax) + ", " + str(b_ay) + ", " + str(mainhand) + ");"
	# # i_command = "insert into " + move + "(id, p_distance, p_angle, m_distance, m_angle, mainhand) " + value
	s_command = "select * from " + moveName + ";"

	# #print (value)
	# #print (i_command)
	# #print (s_command)
	# # cur.execute(i_command)
	cur.execute(s_command)

	#データベースから抽出したムーヴデータ格納
	# move = [[0 for column in range(6)]for row in range(5)] #rowの方のrange(5) ＝　データ数5個分のスペースを確保
	move = list()
	#データベースの中身
	#move = [id, p_distance, p_angle, m_distance, m_angle, mainhand]

	rangeMagn = [1, 1, 1, 1, 1]

	# i_max = 0
	for i, moveInfo in enumerate(cur.fetchall()):
		# print(moveInfo)
		id = moveInfo[0]     #id
		p_distance = moveInfo[1] * (pHeight/standardHeight) * rangeMagn[i] # * (imgPixel/basePixel) #p_distance
		m_distance = moveInfo[3] * (pHeight/standardHeight) * rangeMagn[i] # * (imgPixel/basePixel) #m_distance
		# if count == 5:
		# 	m_distance = m_distance * 1.73
		mainhand = moveInfo[5]  #mainhand
		if mainhand == 2:
			mainhand = 0

		if mainhand == hand:	#listGenerateにコピペするときmoveInfo[5] == 1:
			p_angle = moveInfo[2]
			m_angle = moveInfo[4]
		#movesの左主体ムーヴを右主体ムーヴに変換
		else:
			if moveInfo[2] <= 0: #p_angle
				p_angle = -180.0 - moveInfo[2]
			else:
				p_angle = 180.0 - moveInfo[2]
			if moveInfo[4] <= 0: #m_angle
				m_angle = -180.0 - moveInfo[4]
			else:
				m_angle = 180.0 - moveInfo[4]
			mainhand = hand #mainhand #listGenerateにコピペするときmainhand = 1:
		# i_max = i+1
		move.append([id, p_distance, p_angle, m_distance, m_angle, mainhand])
		# print('i_max', i_max)
		# moves.append([id, p_distance, p_angle, m_distance, m_angle, mainhand])
	# #空要素の配列を削除
	# move = list(filter(lambda lstx:lstx[0]!=0,move))
	#値確認
	# print(move)

	#ムーヴデータ取得
	#rawムーヴ範囲算出
	movePoint = list()

	for m in move:
		#データを座標に変換
		movePoint.append([math.cos(m[4] * math.pi/180) * m[3], math.sin(m[4] * math.pi/180) * m[3]])
	# print("0:", movePoint)
	movePoint = BaseAngleSort(movePoint)
	# print("1:", movePoint)
	for i in range(len(movePoint)-1):
		x = list()
		y = list()
		if movePoint[i][1] < movePoint[i+1][1]:
			x.append(movePoint[i][0])
			x.append(movePoint[i+1][0])
			y.append(movePoint[i][1])
			y.append(movePoint[i+1][1])
		else:
			x.append(movePoint[i+1][0])
			x.append(movePoint[i][0])
			y.append(movePoint[i+1][1])
			y.append(movePoint[i][1])

		Inclination = inclination(x[0], y[0], x[1], y[1])

		p1 = np.array([0.0,0.0])
		p2 = np.array([0.0,0.0])
		p3 = np.array([0.0,0.0])
		p4 = np.array([0.0,0.0])

		#データ1とデータ2のどちらが上にあるか判別しないといけない
		#データ1が上という前提
		allowRange = 15
		if(Inclination < 0):
			p1 = np.array([x[0]+allowRange, y[0]+allowRange])
			p2 = np.array([x[0]-allowRange, y[0]-allowRange])
			p3 = np.array([x[1]-allowRange, y[1]-allowRange])
			p4 = np.array([x[1]+allowRange, y[1]+allowRange])
		elif(Inclination > 0):
			p1 = np.array([x[0]+allowRange, y[0]-allowRange])
			p2 = np.array([x[0]-allowRange, y[0]+allowRange])
			p3 = np.array([x[1]-allowRange, y[1]+allowRange])
			p4 = np.array([x[1]+allowRange, y[1]-allowRange])
		elif(Inclination == 0):
			p1 = np.array([x[0]+allowRange, y[0]])
			p2 = np.array([x[0]-allowRange, y[0]])
			p3 = np.array([x[1]+allowRange, y[1]])
			p4 = np.array([x[1]-allowRange, y[1]])
		else:
			p1 = np.array([x[0], y[0]+allowRange])
			p2 = np.array([x[0], y[0]-allowRange])
			p3 = np.array([x[1], y[1]+allowRange])
			p4 = np.array([x[1], y[1]-allowRange])
		moveRawRange.append([moveName, p1, p2, p3, p4])
		# if moveName == "deadpoint":
		# 	print(p1, p2, p3, p4)
	#ムーヴ後座標
	# x = np.arange(i_max, dtype = 'float64')
	# y = np.arange(i_max, dtype = 'float64')
	# for m in move:
		# x.append(math.cos(m[4] * math.pi/180) * m[3])
		# y.append(math.sin(m[4] * math.pi/180) * m[3])
	# x = [x_ for y_, x_ in sorted(zip(y, x))]
	# y.sort()
	# print(x, y)
	#
	# p1 = np.array([0.0,0.0])
	# p2 = np.array([0.0,0.0])
	# p3 = np.array([0.0,0.0])
	# p4 = np.array([0.0,0.0])
	#
	# Inclination = inclination(x[0], y[0], x[1], y[1])
	#
	# #データ1とデータ2のどちらが上にあるか判別しないといけない
	# #データ1が上という前提
	# allowRange = 15
	# if(Inclination < 0):
	# 	p1 = np.array([x[0]+allowRange, y[0]+allowRange])
	# 	p2 = np.array([x[0]-allowRange, y[0]-allowRange])
	# 	p3 = np.array([x[1]-allowRange, y[1]-allowRange])
	# 	p4 = np.array([x[1]+allowRange, y[1]+allowRange])
	# elif(Inclination > 0):
	# 	p1 = np.array([x[0]+allowRange, y[0]-allowRange])
	# 	p2 = np.array([x[0]-allowRange, y[0]+allowRange])
	# 	p3 = np.array([x[1]-allowRange, y[1]+allowRange])
	# 	p4 = np.array([x[1]+allowRange, y[1]-allowRange])
	# elif(Inclination == 0):
	# 	p1 = np.array([x[0]+allowRange, y[0]])
	# 	p2 = np.array([x[0]-allowRange, y[0]])
	# 	p3 = np.array([x[1]+allowRange, y[1]])
	# 	p4 = np.array([x[1]-allowRange, y[1]])
	# else:
	# 	p1 = np.array([x[0], y[0]+allowRange])
	# 	p2 = np.array([x[0], y[0]-allowRange])
	# 	p3 = np.array([x[1], y[1]+allowRange])
	# 	p4 = np.array([x[1], y[1]-allowRange])
	# # print("p1", p1)
	# # print("p2", p2)
	# # print("p3", p3)
	# # print("p4", p4)
	#
	# #ムーヴ座標格納
	# moveRawRange.append([moveName, p1, p2, p3, p4])
	# moveRawRange.append([moveName, p1, p2, p3, p4, x[0], y[0], x[1], y[1]])
	# #値確認
	# print("Move raw range", moveRawRange)

	# if moveName == "dyno":
	# 	continue
	#rawnonムーヴ範囲算出
	for mP in movePoint:
		nonMoveRawRange.append([mP[0], mP[1]])
	# nonMoveRawRange.append([movePoint[0][0], movePoint[0][1]])
	# nonMoveRawRange.append([movePoint[1][0], movePoint[1][1]])
	# if np.linalg.norm(p1) < np.linalg.norm(p2):
	# 	d14, b14 = line(p1[0], p1[1], p4[0], p4[1])
	# 	d1 = inclination(0, 0, x[0], y[0])
	# 	d4 = inclination(0, 0, x[1], y[1])
	# 	nonMoveRawRange.append(intersection(d14, b14, d1, intercept(d1, x[0], y[0])))
	# 	nonMoveRawRange.append(intersection(d14, b14, d4, intercept(d4, x[1], y[1])))
	# else:
	# 	d23, b23 = line(p2[0], p2[1], p3[0], p3[1])
	# 	d2 = inclination(0, 0, x[0], y[0])
	# 	d3 = inclination(0, 0, x[1], y[1])
	# 	nonMoveRawRange.append(intersection(d23, b23, d2, intercept(d2, x[0], y[0])))
	# 	nonMoveRawRange.append(intersection(d23, b23, d3, intercept(d3, x[1], y[1])))
	# count = count+1

#nonMoveRawRangeを整理
nonMoveRawRange = BaseAngleSort(nonMoveRawRange)
#角度（絶対値）算出
# nonMoveRawAngle = [np.rad2deg(np.arccos(np.clip(np.inner(nmrr, np.array([1.0, 0.0]))/(np.linalg.norm(nmrr) * np.linalg.norm(np.array([1.0, 0.0]))), -1.0, 1.0))) for nmrr in nonMoveRawRange]
# #y軸が負のものの角度にマイナスを付与
# nonMoveRawAngle = [-nmra if nmrr[1] < 0 else nmra for nmra, nmrr in zip(nonMoveRawAngle, nonMoveRawRange)]
# print(nonMoveRawAngle)
# #90度を始めとしてソート
# nMRRA_ = list()
# nMRRA__ = list()
# for nmrr, nmra in zip(nonMoveRawRange, nonMoveRawAngle):
# 	if nmra < 90.0:
# 		nMRRA_.append([nmrr, nmra])
# 	else:
# 		nMRRA__.append([nmrr, nmra])
# # print(str(nMRRA[3]))
# # print(str(nMRRA[4]))
# # nMRRA__ = [nmrra for nmrra in nMRRA if nmrra[1]>=90.0]
# # nMRRA_ = [nmrra for nmrra in nMRRA if nmrra[1]<90.0]
# nMRRA_.sort(key=operator.itemgetter(1))
# nMRRA__.sort(key=operator.itemgetter(1))
# # print("1", nMRRA_)
# # print("2", nMRRA__)
# nonMoveRawRange = [nmrr for nmrr, nmra in nMRRA_] + [nmrr for nmrr, nmra in nMRRA__]
cur.close()
# conn.commit()
conn.close()

os.makedirs(os.path.join("output", "range", str(pHeight)), exist_ok=True)
print(len(nonMoveRawRange))
#練習したいムーヴのムーヴ情報を取り出す
moveData = list()
for mRR in moveRawRange:
	if mRR[0] in practiceMove:
		moveData.append(mRR)
# print("Move data", moveData)

#ムーヴ範囲描画
moveImg = HLowSV(wallImgPath)
for mD in moveData:
	print("moveData:", mD)
	moveImg = cv2.line(moveImg, (int(start[0]+mD[1][0]), int(start[1]+mD[1][1])), (int(start[0]+mD[2][0]), int(start[1]+mD[2][1])), (0, 170, 0), lineSize)
	moveImg = cv2.line(moveImg, (int(start[0]+mD[2][0]), int(start[1]+mD[2][1])), (int(start[0]+mD[3][0]), int(start[1]+mD[3][1])), (0, 170, 0), lineSize)
	moveImg = cv2.line(moveImg, (int(start[0]+mD[3][0]), int(start[1]+mD[3][1])), (int(start[0]+mD[4][0]), int(start[1]+mD[4][1])), (0, 170, 0), lineSize)
	moveImg = cv2.line(moveImg, (int(start[0]+mD[4][0]), int(start[1]+mD[4][1])), (int(start[0]+mD[1][0]), int(start[1]+mD[1][1])), (0, 170, 0), lineSize)

# #ムーヴ後座標の描画
# moveImg = cv2.circle(moveImg, (int(start[0]+moveData[5]),int(start[1]+moveData[6])), circle1Size, (255, 0, 255), -1)
# moveImg = cv2.circle(moveImg, (int(start[0]+moveData[7]),int(start[1]+moveData[8])), circle1Size, (255, 0, 255), -1)

#始点の描画
moveImg = cv2.circle(moveImg, (int(start[0]), int(start[1])), circle2Size, (0, 255, 0), -1)

cv2.imshow("MoveHoldRange", cv2.resize(moveImg, (int(moveImg.shape[1]/2), int(moveImg.shape[0]/2))))

#nonムーヴ範囲描画
nonMoveImg = HLowSV(wallImgPath)
# nonMoveImg = cv2.imread("./resource/x.png")
#移動範囲の描画
# nonMoveImg = cv2.line(nonMoveImg, (int(start[0]), int(start[1])), (int(start[0]+nonMoveRawRange[0][0]), int(start[1]+nonMoveRawRange[0][1])), (0, 170, 0), lineSize)
for i in range(len(nonMoveRawRange)-1):
	# nonMoveImg = cv2.line(nonMoveImg, (int(start[0]+nonMoveRawRange[i][0]), int(start[1]+nonMoveRawRange[i][1])), (int(start[0]+nonMoveRawRange[i+1][0]), int(start[1]+nonMoveRawRange[i+1][1])), (i*30, 0, 0), lineSize)
# nonMoveImg = cv2.line(nonMoveImg, (int(start[0]+nonMoveRawRange[i+1][0]), int(start[1]+nonMoveRawRange[i+1][1])), (int(start[0]+nonMoveRawRange[0][0]), int(start[1]+nonMoveRawRange[0][1])), (0, 0, 0), lineSize)
	nonMoveImg = cv2.line(nonMoveImg, (int(start[0]+nonMoveRawRange[i][0]), int(start[1]+nonMoveRawRange[i][1])), (int(start[0]+nonMoveRawRange[i+1][0]), int(start[1]+nonMoveRawRange[i+1][1])), (0, 153, 255), lineSize)
nonMoveImg = cv2.line(nonMoveImg, (int(start[0]+nonMoveRawRange[i+1][0]), int(start[1]+nonMoveRawRange[i+1][1])), (int(start[0]+nonMoveRawRange[0][0]), int(start[1]+nonMoveRawRange[0][1])), (0, 153, 255), lineSize)
# nonMoveImg = cv2.line(nonMoveImg, (int(start[0]+nonMoveRawRange[i+1][0]), int(start[1]+nonMoveRawRange[i+1][1])), (int(start[0]), int(start[1])), (0, 170, 0), lineSize)
#始点の描画
nonMoveImg = cv2.circle(nonMoveImg, (int(start[0]), int(start[1])), circle2Size, (0, 255, 0), -1)
cv2.imshow("nonMoveHoldRange", cv2.resize(nonMoveImg, (int(nonMoveImg.shape[1]/2), int(nonMoveImg.shape[0]/2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join("output", "range", str(pHeight),  practiceMove+"_"+str(StartHoldNo)+"_"+str(hand)+".png"), moveImg)
cv2.imwrite(os.path.join("output", "range", str(pHeight), "nonMove"+"_"+str(StartHoldNo)+"_"+str(hand)+".png"), nonMoveImg)

#ウォール画像の拡張子無しファイル名
imgBaseName = os.path.basename(wallImgPath).split(".")[0]
#リストのファイル名
moveHoldListName = os.path.join("resource", "list", str(pHeight), practiceMove+'List_'+imgBaseName)
nonMoveHoldListName = os.path.join("resource", "list", str(pHeight), 'nonMoveHoldList_'+imgBaseName)

#----------移動可能ホールドリスト取得
if os.path.isfile(moveHoldListName):
	moveHoldList = joblib.load(moveHoldListName)
else:
	sys.exit("Don't find list file! Please run listGenerate.py")
if os.path.isfile(nonMoveHoldListName):
	nonMoveHoldList = joblib.load(nonMoveHoldListName)
else:
	sys.exit("Don't find list file! Please run listGenerate.py")
# print(moveHoldList)
# print(moveHoldList[hand])
# print(len(moveHoldList[hand][0]))

# for hand in range(2):
# 	print(hand)
# 	for i, (mhl, nmhl) in enumerate(zip(moveHoldList[hand], nonMoveHoldList[hand])):
# 		print(i, len(mhl), len(nmhl))

#取得ホールドの描画
rangeImg = HLowSV(wallImgPath)
for hp in holdPoint:
	rangeImg = cv2.circle(rangeImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 220, 255), -1)
#nonmoveホールドの描画
for nmh in nonMoveHoldList[hand][holdPoint.index(start)]:
		rangeImg = cv2.circle(rangeImg, (int(nmh[0]), int(nmh[1])), circle1Size, (255, 0, 0), -1)
#moveホールドの描画
for mh in moveHoldList[hand][holdPoint.index(start)]:
		rangeImg = cv2.circle(rangeImg, (int(mh[0]), int(mh[1])), circle1Size, (0, 255, 0), -1)
# 始点の描画
rangeImg = cv2.circle(rangeImg, (int(start[0]), int(start[1])), circle1Size, (0, 0, 255), -1)
cv2.imshow("range"+"_"+str(StartHoldNo)+"_"+str(hand), cv2.resize(rangeImg, (int(rangeImg.shape[1]/2), int(rangeImg.shape[0]/2))))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join("output", "range",str(pHeight) , "hold_"+practiceMove+str(StartHoldNo)+"_"+str(hand)+".png"), rangeImg)
