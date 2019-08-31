#python listGenerate.py wallImgPath csvPath Height
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
import time
from Bmodules import resizer

outputShape = (831, 1108)
resizeShape = (1500, 2700)
#基準身長
standardHeight = 165.0
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

#距離角度を座標に変換
def PointCalc(move):
	px = math.cos(move[2]*math.pi/180) * move[1]
	py = math.sin(move[2]*math.pi/180) * move[1]
	mx = math.cos(move[4]*math.pi/180) * move[3]
	my = math.sin(move[4]*math.pi/180) * move[3]
	return px, py, mx, my
#座標を角度順にソート
def AngleSort(basePoint, targetPoint, pointList):
	baseVec = np.array([targetPoint[0]-basePoint[0], targetPoint[1]-basePoint[1]])
	Angle = [np.rad2deg(np.arccos(np.clip(np.inner(baseVec, np.array([pl[0]-basePoint[0], pl[1]-basePoint[1]]))/(np.linalg.norm(baseVec) * np.linalg.norm(np.array([pl[0]-basePoint[0], pl[1]-basePoint[1]]))), -1.0, 1.0))) for pl in pointList]
	return [pl for ang, pl in sorted(zip(Angle, pointList))]

#座標と(0.0, 1.0)角度でソート
def BaseAngleSort(pointList):
	baseLine = np.array([0.0, 1.0])
	#角度（絶対値）算出
	angleList = [np.rad2deg(np.arccos(np.clip(np.inner(pl, np.array([0.0, 1.0]))/(np.linalg.norm(pl) * np.linalg.norm(np.array([0.0, 1.0]))), -1.0, 1.0))) for pl in pointList]
	#x軸が正のものの角度を90度を始めとした角度になるよう計算
	angleList = [360-al if pl[0] > 0 else al for al, pl in zip(angleList, pointList)]

	return [pl for al, pl in sorted(zip(angleList, pointList))]

#移動可能なホールドの取得
def getNonMoveRangeHold(targetHp, rawData, hpList, hand):
	nmP = list()
	# nmP.append(targetHp)
	for nmrr in rawData:
		if hand == 0:
			nmP.append([float(targetHp[0])-float(nmrr[0]), float(targetHp[1])+float(nmrr[1])])
		else:
			nmP.append([float(targetHp[0])+float(nmrr[0]), float(targetHp[1])+float(nmrr[1])])
	#範囲
	nonMoveHoldRange = Polygon(*nmP)
	# #値確認
	# print("Non move hold range:", *nmP)
	return [hp for hp in hpList if nonMoveHoldRange.encloses_point(hp)]
#ムーブ可能なホールドの取得
def getMoveRangeHold(targetHp, rawData, hpList, hand):
	mP = list()
	# print("1",targetHp)
	# print("3",rawData)
	if hand == 0:# 左手
		# for rD in rawData:
		# 	mP.append([targetHp[0]-rD[0], targetHp[1]+rD[1])
		mP.append([float(targetHp[0])-float(rawData[1][0]), float(targetHp[1])+float(rawData[1][1])])
		mP.append([float(targetHp[0])-float(rawData[2][0]), float(targetHp[1])+float(rawData[2][1])])
		mP.append([float(targetHp[0])-float(rawData[3][0]), float(targetHp[1])+float(rawData[3][1])])
		mP.append([float(targetHp[0])-float(rawData[4][0]), float(targetHp[1])+float(rawData[4][1])])
	else:# 右手
		# for rD in rawData:
		# 	mP.append([targetHp[0]+rD[0], targetHp[1]+rD[1])
		mP.append([float(targetHp[0])+float(rawData[1][0]), float(targetHp[1])+float(rawData[1][1])])
		mP.append([float(targetHp[0])+float(rawData[2][0]), float(targetHp[1])+float(rawData[2][1])])
		mP.append([float(targetHp[0])+float(rawData[3][0]), float(targetHp[1])+float(rawData[3][1])])
		mP.append([float(targetHp[0])+float(rawData[4][0]), float(targetHp[1])+float(rawData[4][1])])
	#範囲
	MoveHoldRange = Polygon(*mP)
	#値確認
	# print("Move hold range:", *mP)
	return [hp for hp in hpList if MoveHoldRange.encloses_point(hp)]

if __name__ == '__main__':
	#ムーヴ情報
	moves = ["deadpoint", "crossmove", "diagonal", "dropknee", "dyno"]
	#0を原点としたmove情報
	moveRawRange = list()
	#moveRawRange = [[ムーヴ名, 原点に近いp1, 原点に近いp2]]
	#0を原点としたnonmove情報
	nonMoveRawRange = list()
	MoveHoldsRange = list()
	#入力
	#move = ["deadpoint","dyagonal"]
	# wallImgPath = "./resource/IMG_9519.JPG"
	wallImgPath = sys.argv[1]
	wallImg = resizer.pixelResize(cv2.imread(wallImgPath), outputShape[0], outputShape[1])
	# wallImg = cv2.resize(wallImg, (outputShape[0], outputShape[1]))
	wallShape = [wallImg.shape[1], wallImg.shape[0]]

	csvResizeX = wallShape[0]/resizeShape[0]
	csvResizeY = wallShape[1]/resizeShape[1]

	#事前入力
	#身長
	# print("Input height")
	# pHeight = float((input(" > ")))
	pHeight = float(sys.argv[3])

	conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='test3_db'
	)

	cur = conn.cursor()

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

			if mainhand == 1:
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
				mainhand = 1 #mainhand
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

	#事前入力
	# #ムーヴ
	# print("Select move")
	# print("deadpoint:1 crossmove:2 diagonal:3 dropknee:4 dyno:5")
	# # practiceMove = int(sys.argv[1])
	# practiceMove = int(input(" > "))
	# if practiceMove == 1:
	# 	print("Select deadpoint")
	# 	practiceMove = "deadpoint"
	# elif practiceMove == 2:
	# 	print("Select crossmove")
	# 	practiceMove = "crossmove"
	# elif practiceMove == 3:
	# 	print("Select diagonal")
	# 	practiceMove = "diagonal"
	# elif practiceMove == 4:
	# 	print("Select dropknee")
	# 	practiceMove = "dropknee"
	# elif practiceMove == 5:
	# 	print("Select dyno")
	# 	practiceMove = "dyno"
	# else:
	# 	print("Input error", practiceMove)
	# 	sys.exit()

	os.makedirs(os.path.join("resource", "list", str(pHeight)), exist_ok=True)

	#ウォール画像の拡張子無しファイル名
	imgBaseName = os.path.basename(wallImgPath).split(".")[0]


	# print(nonMoveHoldListName)
	#----------マップ生成開始
	#非ムーヴ移動ホールドリストの生成
	#リストのファイル名
	nonMoveHoldListName = os.path.join("resource", "list", str(pHeight), 'nonMoveHoldList_'+imgBaseName)
	if os.path.isfile(nonMoveHoldListName):
		print(nonMoveHoldListName+" is existed")
	else:
		print("Generating non move hold list...")
		nonMoveHoldList = [[],[]]
		nonMoveHoldList[0] = joblib.Parallel(n_jobs=-1, verbose=7)([joblib.delayed(getNonMoveRangeHold)(hold, nonMoveRawRange, holdPoint, 0) for hold in holdPoint])
		nonMoveHoldList[1] = joblib.Parallel(n_jobs=-1, verbose=7)([joblib.delayed(getNonMoveRangeHold)(hold, nonMoveRawRange, holdPoint, 1) for hold in holdPoint])
		joblib.dump(nonMoveHoldList, nonMoveHoldListName, compress=3)
		print("Complete generate! File name is "+nonMoveHoldListName)
		del nonMoveHoldList

	#ムーヴ移動ホールドリストの生成
	for practiceMove in moves:
		#練習したいムーヴのムーヴ情報を取り出す
		moveData = list()
		for mRR in moveRawRange:
			if mRR[0] in practiceMove:
				moveData.append(mRR)
		# print("Move data", moveData)

		#リストのファイル名
		moveHoldListName = os.path.join("resource", "list", str(pHeight), practiceMove+'List_'+imgBaseName)

		if os.path.isfile(moveHoldListName):
			print(moveHoldListName+" is existed")
		else:
			print("Generating "+practiceMove+" hold list...")
			moveHoldList = [[],[]]
			for mD in moveData:
				moveHoldList[0].extend(joblib.Parallel(n_jobs=-1, verbose=7)([joblib.delayed(getMoveRangeHold)(hold, mD, holdPoint, 0) for hold in holdPoint]))
				moveHoldList[1].extend(joblib.Parallel(n_jobs=-1, verbose=7)([joblib.delayed(getMoveRangeHold)(hold, mD, holdPoint, 1) for hold in holdPoint]))
			joblib.dump(moveHoldList, moveHoldListName, compress=3)
			print("Complete generate! File name is "+moveHoldListName)
			del moveHoldList
	# #値確認
	# print("leftMoveHoldList", moveHoldList[0])
	# print("leftNonMoveHoldList", nonMoveHoldList[0])
	# print("rightMoveHoldList", moveHoldList[1])
	# print("rightNonMoveHoldList", nonMoveHoldList[1])
	#----------マップ生成終了
