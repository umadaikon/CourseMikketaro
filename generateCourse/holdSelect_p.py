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
from joblib import Parallel, delayed
import pickle
import os
from Bmodules import resizer
import random

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

#ムーブ可能なホールドの取得
def getMoveRangeHold(targetHp, rawData, hpList, hand):
	mP = list()
	if hand == 0:# 左手
		mP.append([targetHp[0]-rawData[1][0], targetHp[1]+rawData[1][1]])
		mP.append([targetHp[0]-rawData[2][0], targetHp[1]+rawData[2][1]])
		mP.append([targetHp[0]-rawData[3][0], targetHp[1]+rawData[3][1]])
		mP.append([targetHp[0]-rawData[4][0], targetHp[1]+rawData[4][1]])
	else:# 右手
		mP.append([targetHp[0]+rawData[1][0], targetHp[1]+rawData[1][1]])
		mP.append([targetHp[0]+rawData[2][0], targetHp[1]+rawData[2][1]])
		mP.append([targetHp[0]+rawData[3][0], targetHp[1]+rawData[3][1]])
		mP.append([targetHp[0]+rawData[4][0], targetHp[1]+rawData[4][1]])
	#範囲
	MoveHoldRange = Polygon(*mP)
	#値確認
	# print("Move hold range:", *mP)
	return [hp for hp in hpList if MoveHoldRange.encloses_point(hp)]
#移動可能なホールドの取得
def getNonMoveRangeHold(targetHp, rawData, hpList, hand):
	nmP = list()
	nmP.append(targetHp)
	for nmrr in rawData:
		if hand == 0:
			nmP.append([nmP[0][0]-nmrr[0], nmP[0][1]+nmrr[1]])
		else:
			nmP.append([nmP[0][0]+nmrr[0], nmP[0][1]+nmrr[1]])
	#範囲
	nonMoveHoldRange = Polygon(*nmP)
	# #値確認
	# print("Non move hold range:", *nmP)
	return [hp for hp in hpList if nonMoveHoldRange.encloses_point(hp)]

#遡るホールドを検索し、インデックスを返す
def SearchBackIndex(courseList, stack, hand):
	try:
		return courseList.index([stack[3][0], stack[3][1], 0])
	except ValueError:
		try:
			return courseList.index([stack[3][0], stack[3][1], 1])
		except ValueError:
			try:
				return courseList.index([stack[3][0], stack[3][1]])
			except ValueError:
				return -1
	sys.exit("Don't find back hold")

#ホールド座標からホールド番号を返す
def printHoldNo(holdList, hP):
	getNo = list()
	for hl in holdList:
		getNo.append(hP.index(hl[:2]))
	return getNo
#親ホールドの番号も返す
def printHoldHoldNo(holdList, hP):
	getNo = list()
	for hl in holdList:
		if hl[3] == [-1, -1]:
			getNo.append(hP.index(hl[:2]))
		else:
			getNo.append([hP.index(hl[:2]), hP.index(hl[3])])
	return getNo

#重複した要素を削除
def get_unique_list(sequence):
	seen = []
	for s in sequence:
		if s not in seen:
			seen.append(s)
	return seen

if __name__ == '__main__':
	#ムーヴ情報
	moves = list()
	#0を原点としたmove情報
	moveRawRange = list()
	#moveRawRange = [[ムーヴ名, 原点に近いp1, 原点に近いp2]]
	#0を原点としたnonmove情報
	nonMoveRawRange = list()
	MoveHoldsRange = list()

	#入力
	#move = ["deadpoint","dyagonal"]
	moveName = ""
	count = 1
	wallImgPath = sys.argv[1]
	# wallImg = resizer.pixelResize(cv2.imread(wallImgPath), outputShape[0], outputShape[1])
	# wallImg = cv2.resize(wallImg, (outputShape[0], outputShape[1]))
	wallImg = HLowSV(wallImgPath)
	wallShape = [wallImg.shape[1], wallImg.shape[0]]
	csvResizeX = wallShape[0]/resizeShape[0]
	csvResizeY = wallShape[1]/resizeShape[1]

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
	#始点
	start = list()
	#終点
	goal = list()
	#ムーヴ可能ホールドリスト
	moveHoldList = list()
	#移動可能ホールドリスト
	nonMoveHoldList = list()

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

	resizeWallImg = cv2.resize(wallImg, (int(wallImg.shape[1]*(2/3)), int(wallImg.shape[0]*(2/3))))
	#事前入力
	#ユーザ名
	# print("Input your name")
	# user = input(" > ")
	user = sys.argv[3]
	#身長
	# print("Input height")
	# pHeight = float((input(" > ")))
	pHeight = float(sys.argv[4])
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
	#始点、終点
	print("Select start hold")
	print("Left hold")
	# lStartHoldNo = sys.argv[2]
	cv2.imshow("Select hold", resizeWallImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	lStartHoldNo = input("holdNo > ")
	print("Left start", holdPoint[int(lStartHoldNo)])
	start.append([holdPoint[int(lStartHoldNo)][0], holdPoint[int(lStartHoldNo)][1], 0, [-1, -1]])
	print("Right hold")
	# rStartHoldNo = sys.argv[3]
	cv2.imshow("Select hold", resizeWallImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	rStartHoldNo = input("holdNo > ")
	print("Right start", holdPoint[int(rStartHoldNo)])
	start.append([holdPoint[int(rStartHoldNo)][0], holdPoint[int(rStartHoldNo)][1], 0, [-1, -1]])
	print("Select goal hold")
	# goalHoldNo = sys.argv[4]
	cv2.imshow("Select hold", resizeWallImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	goalHoldNo = input("holdNo > ")
	print("Goal", holdPoint[int(goalHoldNo)])
	goal = holdPoint[int(goalHoldNo)]

	os.makedirs(os.path.join("D:", "course", user, "deadpoint"), exist_ok=True)
	os.makedirs(os.path.join("D:", "course", user, "crossmove"), exist_ok=True)
	os.makedirs(os.path.join("D:", "course", user, "diagonal"), exist_ok=True)
	os.makedirs(os.path.join("D:", "course", user, "dropknee"), exist_ok=True)
	os.makedirs(os.path.join("D:", "course", user, "dyno"), exist_ok=True)

	# #練習したいムーヴのムーヴ情報を取り出す
	# for mRR in moveRawRange:
	# 	if mRR[0] in practiceMove:
	# 		moveData = mRR
	# print("Move data", moveData)

	#ウォール画像の拡張子無しファイル名
	imgBaseName = os.path.basename(wallImgPath).split(".")[0]
	#リストのファイル名
	moveHoldListName = os.path.join("resource", "list", str(pHeight), practiceMove+'List_'+imgBaseName+'.txt')
	nonMoveHoldListName = os.path.join("resource", "list", str(pHeight), 'nonMoveHoldList_'+imgBaseName+'.txt')

	#----------移動可能ホールドリスト取得
	if os.path.isfile(moveHoldListName):
		# print("Get "+moveHoldListName)
		with open(moveHoldListName, 'rb') as fm:
			moveHoldList = pickle.load(fm)
	else:
		sys.exit("Don't find list file! Please run listGenerate.py")
	if os.path.isfile(nonMoveHoldListName):
		# print("Get "+nonMoveHoldListName)
		with open(nonMoveHoldListName, 'rb') as fnm:
			nonMoveHoldList = pickle.load(fnm)
	else:
		sys.exit("Don't find list file! Please run listGenerate.py")

	#-----------コース生成開始
	print("Start generate course")
	#コース
	course = list()
	course = [[], []]
	def GenerateCourse(hand):
	# for hand in range(2):#hand==0 左手 hand==1 右手
		#生成されたコースの数をカウント
		courseCount = list()
		#生成途中のコース
		courseHold = list()
		#コースの数
		courseCount = int()
		#探索するホールドのスタック
		stackHold = list()
		#stackHold = [[ホールドx座標, ホールドy座標, [移動前のホールド座標], if ムーヴホールド]]
		#既に探索したホールド
		moveCheckedHold = list()
		nonMoveCheckedHold = list()
		checkedHold = list()

		#変数代入
		courseCount = 0
		stackHold.append(start[hand])
		# if hand == 0:
		# 	B = 255
		# 	point = -7
		# elif hand == 1:
		# 	B = 0
		# 	point = 2

		while(not stackHold == []):
			P = stackHold[-1]
			courseHold.append(P[:3])
			# if P[2] == 1 and P[:2] not in moveCheckedHold:
			# 	moveCheckedHold.append(P[:3])
			# if P[:2] not in nonMoveCheckedHold:
			# 	nonMoveCheckedHold.append(P[:3])
			if not P[:2] in checkedHold:
				checkedHold.append(P[:2])
			# #値確認
			# print()
			# print("Stack", printHoldHoldNo(stackHold, holdPoint))
			# print("Pop", printHoldHoldNo([stackHold[-1]], holdPoint))
			# print()
			# print("Course hold", printHoldNo(courseHold, holdPoint))
			del stackHold[-1]
			if courseHold[-1][:2] == [goal[0], goal[1]]:
				# #値確認
				# print("Find course", courseCount, "Hand", hand)
				# input()
				course[hand].append(copy.deepcopy(courseHold))
				if not stackHold == []:
					backHoldIndex = SearchBackIndex(courseHold, stackHold[-1], hand)
					while backHoldIndex == -1 and not stackHold == []:
						del stackHold[-1]
						if not stackHold == []:
							backHoldIndex = SearchBackIndex(courseHold, stackHold[-1], hand)
					if not stackHold == []:
						# #値確認
						# print("Del", printHoldNo(holdPoint, courseHold[backHoldIndex+1:]))
						del courseHold[backHoldIndex+1:]
				if not stackHold == []:
					backHoldIndex = SearchBackIndex(courseHold, stackHold[-1], hand)
					del courseHold[backHoldIndex+1:]
				#確認不要ホールドリストの再構築
				# moveCheckedHold = [mh for mh in courseHold if mh[2] == 1]
				# nonMoveCheckedHold = [nmh for nmh in courseHold]
				checkedHold = copy.deepcopy(courseHold)
				if not len(courseHold) == 1:
					for ch in courseHold[:-1]:
						checkedHold.extend(copy.deepcopy(moveHoldList[hand][holdPoint.index(ch[:2])]))
						checkedHold.extend(copy.deepcopy(nonMoveHoldList[hand][holdPoint.index(ch[:2])]))
				checkedHold = get_unique_list(checkedHold)
				courseCount = courseCount+1
				continue
			#-----------ムーヴホールド確認
			MoveHold = moveHoldList[hand][holdPoint.index(P[:2])]
			# MoveHold = moveHoldList[hand][holdPoint.index(P[:2])]
			# #値確認
			# print("Movehold", printHoldNo(MoveHold, holdPoint))
			# print("Checked hold", printHoldNo(checkedHold, holdPoint))
			# print("Checked(Move)", moveCheckedHold)

			for i, mh in reversed(list(enumerate(MoveHold))):
				# for checked in moveCheckedHold:
				for checked in checkedHold:
					if mh == checked[:2]:
						del MoveHold[i]
						break
			# #値確認
			# print("Checked movehold", printHoldNo(MoveHold, holdPoint))

			if not MoveHold == []:
				MoveHold = AngleSort(P[:2], goal, MoveHold)
			#-----------ムーヴホールド確認

			#-----------nonムーヴホールド確認
			nonMoveHold = nonMoveHoldList[hand][holdPoint.index(P[:2])]
			# #値確認
			# print("Nonmovehold", printHoldNo(nonMoveHold, holdPoint))
			# print("Checked hold", printHoldNo(checkedHold, holdPoint))
			# print("Checked(nonMove)", nonMoveCheckedHold)

			for i, nmh in reversed(list(enumerate(nonMoveHold))):
				# for checked in nonMoveCheckedHold:
				for checked in checkedHold:
					if nmh == checked[:2]:
						del nonMoveHold[i]
						break
			# #値確認
			# print("Checked nonmoveHold", printHoldNo(nonMoveHold, holdPoint))

			if not nonMoveHold == []:
				nonMoveHold = AngleSort(P[:2], goal, nonMoveHold)
			#----------nonムーヴホールド確認

			#何も見つからなかった場合コースを必要分遡る
			if nonMoveHold == [] and MoveHold == [] and not stackHold == []:
				backHoldIndex = SearchBackIndex(courseHold, stackHold[-1], hand)
				while backHoldIndex == -1 and not stackHold == []:
					del stackHold[-1]
					if not stackHold == []:
						backHoldIndex = SearchBackIndex(courseHold, stackHold[-1], hand)
				if not stackHold == []:
					# #値確認
					# print("Del", printHoldNo(courseHold[backHoldIndex+1:], holdPoint))
					for ch in courseHold[backHoldIndex+1:]:
						for nmhl in nonMoveHoldList[hand][holdPoint.index(ch[:2])]:
							if nmhl in checkedHold:
								checkedHold.remove(nmhl)
						for mhl in moveHoldList[hand][holdPoint.index(ch[:2])]:
							if mhl in checkedHold:
								checkedHold.remove(mhl)
					del courseHold[backHoldIndex+1:]

			# print()
			#stackに追加
			if not nonMoveHold == []:
				for nmh in reversed(nonMoveHold):
					stackHold.append([nmh[0], nmh[1], 0, P[:2]])
				# #値確認
				# print("Append non", nonMoveHold)
				nonMoveHold.clear()
			if not MoveHold == []:
				for mh in reversed(MoveHold):
					stackHold.append([mh[0], mh[1], 1, P[:2]])
				# #値確認
				# print("Append", MoveHold)
				MoveHold.clear()

			#移動可能範囲のホールドをコースの候補に含めない
			checkedHold.extend(copy.deepcopy(moveHoldList[hand][holdPoint.index(P[:2])]))
			checkedHold.extend(copy.deepcopy(nonMoveHoldList[hand][holdPoint.index(P[:2])]))
			checkedHold = get_unique_list(checkedHold)
		return course[hand]
	course = Parallel(n_jobs=2, verbose=10)([delayed(GenerateCourse)(h) for h in range(2)])
	# print(course)
	#-----------コース生成終了
	# print()
	# print("Course data", course)
	if not course[0] == [] and not course[1] == []:
		print("Left courses", len(course[0]), ", Right courses", len(course[1]))
		print("Number of course is " + str(len(course[0])*len(course[1])))
		print("Reshaping course...")
		#コース整形
		for hand in range(2):
			for i, crs in enumerate(course[hand]):
				# #値確認
				# print()
				# print("before:", printHoldNo(crs, holdPoint))
				courseCheck = [1] * len(crs)
				courseCheck[0] = -1
				hold = crs[0]
				while(not hold == crs[-1]):
					nowIndex = crs.index(hold)
					inRangeHold = list()
					inRangeNmh = list()
					inRangeMh = list()
					findIndex = list()
					# #値確認
					# print("hold",  printHoldNo([hold], holdPoint))
					# print("Current check", courseCheck)
					# print("now", nowIndex)
					if hold[:2] in holdPoint:
						inRangeNmh = nonMoveHoldList[hand][holdPoint.index(hold[:2])]
						inRangeMh = moveHoldList[hand][holdPoint.index(hold[:2])]
					inRangeHold = inRangeNmh + inRangeMh
					# #値確認
					# print("inRange", printHoldNo(inRangeHold, holdPoint))
					# print("nmh", printHoldNo(inRangeNmh, holdPoint))
					# print("mh", printHoldNo(inRangeMh, holdPoint))
					if not inRangeHold == []:
						for irh in inRangeHold:
							# #値確認
							# print("irh", irh)
							if [irh[0], irh[1], 0] in crs:
								# print("find!0")
								findIndex.append(crs.index([irh[0], irh[1], 0]))
							if [irh[0], irh[1], 1] in crs:
								# print("find!1")
								findIndex.append(crs.index([irh[0], irh[1], 1]))
						# print("findIndex 0", findIndex)
						findIndex = sorted(list(set(findIndex)))
						# print("findIndex 1", findIndex)
						#今参照しているホールドより後だけ残す
						findIndex = [fI for fI in findIndex if fI > nowIndex]
						# print("findIndex 2", findIndex)
						#今参照しているホールドから始める
						if len(findIndex) >= 2:
							#findIndexのはじめにある番号から最後にある番号までに該当するcourseCheckを0にする
							courseCheck = np.array(courseCheck)
							courseCheck[findIndex[0]:findIndex[-1]] = 0
							courseCheck = list(courseCheck)
							# #findIndexを[0]から探索し、連続しなくなったインデックスを記録
							# if findIndex[0] == nowIndex+1:
							# 	serial = 0
							# 	for i in range(len(findIndex)-1):
							# 		# print(findIndex[i], findIndex[i+1])
							# 		if findIndex[i]+1 == findIndex[i+1]:
							# 			serial = serial + 1
							# 		else:
							# 			break
							# 	#連続しなくなったインデックスの前までを残す
							# 	findIndex = findIndex[:serial+1]
							# 	# print("findIndex 3", findIndex)
							# 	#findIndex内にある番号をインデックスとし、courseCheckの要素を0に変更する
							# 	if len(findIndex) >= 2:
							# 		for fI in findIndex:
							# 			courseCheck[fI] = 0
							# 		courseCheck[len(crs)-list(reversed(courseCheck)).index(0)-1] = 1
					courseCheck[-1] = 1
					# print(nowIndex + courseCheck[nowIndex+1:].index(1))
					# print(hold)
					hold = crs[nowIndex+1 + courseCheck[nowIndex+1:].index(1)]
				course[hand][i] = [hold for hold, cC in zip(crs, courseCheck) if cC == 1 or cC == -1]

				# #値確認
				# print("afterC:", courseCheck)
				# print("afterH:", printHoldNo(course[hand][i], holdPoint))
				# input()
		print("Complete reshape")
		print("Deleting overlap courses...")
		#重複コースの削除
		for hand in range(2):
			course[hand] = get_unique_list(course[hand])

		print("Left courses", len(course[0]), ", Right courses", len(course[1]))
		print("Number of course is " + str(len(course[0])*len(course[1])))
		print("Combin left courses and right courses")
		print("Deleting nothing move courses...")
		route = list()
		leftMoveCount = int()
		rightMoveCount = int()
		for r in course[1]:
			rightMoveCount = 0
			for rhold in r:
				if rhold[2] == 1:
					rightMoveCount = rightMoveCount + 1
					if rightMoveCount > 1:
						break
			if rightMoveCount > 1:
				# print("right break")
				continue
			else:
				for l in course[0]:
					# print("route", l, r)
					leftMoveCount = 0
					for lhold in l:
						if lhold[2] == 1:
							leftMoveCount = leftMoveCount + 1
							if leftMoveCount > 1:
								break
					if leftMoveCount > 1:
						# print("left break")
						continue
					if (rightMoveCount == 1 and leftMoveCount == 0) or (rightMoveCount == 0 and leftMoveCount == 1):
						route.append([l, r])


		# routeCheck = [1] * len(route)
		# route_ = list()
		# routeLen = list()
		#
		# for i, rt in enumerate(route):
		# 	lLen = len(rt[0])-1
		# 	rLen = len(rt[1])-1
		# 	routeLen.append(lLen+rLen)
		# 	# print(rt[0])
		# 	# print(rt[1])
		# 	# print("len", lLen+rLen)
		# 	if not (lLen+rLen <= 5 and lLen+rLen >= 3):
		# 		routeCheck[i] = 0
		# for rc in routeCheck:
		# 	if rc == 1:
		# 		route_.append(route[rc])
		# route = route_
		# print(min(routeLen))
		# routeLen = list()
		# for rt in route:
		# 	lLen = len(rt[0])-1
		# 	rLen = len(rt[1])-1
		# 	routeLen.append(lLen+rLen)
		# route = [rt for rtl, rt in sorted(zip(routeLen, route))]

		print("Complete generate " + str(len(route)) + " courses")
		if not len(route) == 0:
			#コースの描画
			print("Draw start?")
			input()
			print("Drawing...")

			courseNo = 0
			outRouteNo = list()
			# outRouteNo = range(len(5))
			# outRouteNo = range(len(route))
			if len(route) > 1000:
				outRouteNo = random.sample(range(len(route)), 1000)
				outRouteNo.sort()
			else:
				outRouteNo = range(len(route))
			# #値確認
			# print(outRouteNo)
			outCourses = list()
			def DrawCourse(i, no):
				# no = random.randint(0, len(route)-1)
				# no = i
				l = route[no][0]
				r = route[no][1]
				# #値確認
				# print("Left", l)
				# print("Right", r)
				# input()
				courseImg = HLowSV(wallImgPath)
				#ホールド座標の描画
				# for j, hp in enumerate(holdPoint):
				# 	courseImg = cv2.circle(courseImg, (int(hp[0]), int(hp[1])), 5, (255, 200, 255), -1)
				#コース線の描画
				for No, lhold in enumerate(l):
					if not No == 0:
						courseImg = cv2.line(courseImg, (int(l[No-1][0]), int(l[No-1][1])), (int(lhold[0]), int(lhold[1])), (255, 102, 0), lineSize)
				#コース線の描画
				for No, rhold in enumerate(r):
					if not No == 0:
						courseImg = cv2.line(courseImg, (int(r[No-1][0]), int(r[No-1][1])), (int(rhold[0]), int(rhold[1])), (0, 0, 255), lineSize)
				#ホールドの描画
				for No, lhold in enumerate(l):
					if lhold[2] == 1:
						# courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle3Size, (204, 153, 0), -1)
						courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle3Size, (0, 192, 255), -1)
					else:
						courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle2Size, (0, 0, 0), -1)
					holdIndex = holdPoint.index(lhold[:2])
					# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
					courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle1Size, (255, 102, 0), -1)
				#ホールドの描画
				for No, rhold in enumerate(r):
					if rhold[2] == 1:
						# courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle3Size, (204, 153, 0), -1)
						courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle3Size, (0, 192, 255), -1)
					else:
						courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 0, 0), -1)
						holdIndex = holdPoint.index(rhold[:2])
					# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
					courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle1Size, (0, 0, 255), -1)
				#コース番号の描画
				for No, lhold in enumerate(l):
					courseImg = cv2.putText(courseImg, str(No), (int(lhold[0]+leftFontPoint), int(lhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 102, 0), fontThick)
				#コース番号の描画
				for No, rhold in enumerate(r):
					courseImg = cv2.putText(courseImg, str(No), (int(rhold[0]+rightFontPoint), int(rhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThick)
				# 始点、終点の描画
				# courseImg = cv2.circle(courseImg, (int(start[0][0]), int(start[0][1])), 3, (0, 255, 255), -1)
				# courseImg = cv2.circle(courseImg, (int(start[1][0]), int(start[1][1])), 3, (0, 255, 255), -1)
				# courseImg = cv2.circle(courseImg, (int(goal[0]), int(goal[1])), 3, (0, 255, 255), -1)
				cv2.imwrite(os.path.join("D:", "course", user, practiceMove, practiceMove+str(i)+".png"), courseImg)
				# cv2.imshow("course"+str(courseNo), courseImg)#cv2.resize(courseImg, (int(courseImg.shape[1]/5), int(courseImg.shape[0]/5))))
				# outCourses.append(courseImg)
			Parallel(n_jobs=-1, verbose=3, backend="threading")([delayed(DrawCourse)(i, no) for i, no in enumerate(outRouteNo)])
			# for no in outRouteNo:
			# # for no, rt in enumerate(route):
			# 	# no = random.randint(0, len(route)-1)
			# 	# no = i
			# 	l = route[no][0]
			# 	r = route[no][1]
			# 	# #値確認
			# 	# print("Left", l)
			# 	# print("Right", r)
			# 	# input()
			# 	courseImg = HLowSV(wallImgPath)
			# 	#ホールド座標の描画
			# 	# for j, hp in enumerate(holdPoint):
			# 	# 	courseImg = cv2.circle(courseImg, (int(hp[0]), int(hp[1])), 5, (255, 200, 255), -1)
			# 	#コース線の描画
			# 	for No, lhold in enumerate(l):
			# 		if not No == 0:
			# 			courseImg = cv2.line(courseImg, (int(l[No-1][0]), int(l[No-1][1])), (int(lhold[0]), int(lhold[1])), (255, 102, 0), lineSize)
			# 	#コース線の描画
			# 	for No, rhold in enumerate(r):
			# 		if not No == 0:
			# 			courseImg = cv2.line(courseImg, (int(r[No-1][0]), int(r[No-1][1])), (int(rhold[0]), int(rhold[1])), (0, 0, 255), lineSize)
			# 	#ホールドの描画
			# 	for No, lhold in enumerate(l):
			# 		if lhold[2] == 1:
			# 			# courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle3Size, (204, 153, 0), -1)
			# 			courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle3Size, (0, 192, 255), -1)
			# 		else:
			# 			courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle2Size, (0, 0, 0), -1)
			# 		holdIndex = holdPoint.index(lhold[:2])
			# 		# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
			# 		courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle1Size, (255, 102, 0), -1)
			# 	#ホールドの描画
			# 	for No, rhold in enumerate(r):
			# 		if rhold[2] == 1:
			# 			# courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle3Size, (204, 153, 0), -1)
			# 			courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle3Size, (0, 192, 255), -1)
			# 		else:
			# 			courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 0, 0), -1)
			# 			holdIndex = holdPoint.index(rhold[:2])
			# 		# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
			# 		courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle1Size, (0, 0, 255), -1)
			# 	#コース番号の描画
			# 	for No, lhold in enumerate(l):
			# 		courseImg = cv2.putText(courseImg, str(No), (int(lhold[0]+leftFontPoint), int(lhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), fontThick)
			# 	#コース番号の描画
			# 	for No, rhold in enumerate(r):
			# 		courseImg = cv2.putText(courseImg, str(No), (int(rhold[0]+rightFontPoint), int(rhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThick)
			# 	# 始点、終点の描画
			# 	# courseImg = cv2.circle(courseImg, (int(start[0][0]), int(start[0][1])), 3, (0, 255, 255), -1)
			# 	# courseImg = cv2.circle(courseImg, (int(start[1][0]), int(start[1][1])), 3, (0, 255, 255), -1)
			# 	# courseImg = cv2.circle(courseImg, (int(goal[0]), int(goal[1])), 3, (0, 255, 255), -1)
			# 	cv2.imwrite(os.path.join("D:", "course", user, practiceMove, practiceMove+str(courseNo)+".png"), courseImg)
			# 	# cv2.imshow("course"+str(courseNo), courseImg)#cv2.resize(courseImg, (int(courseImg.shape[1]/5), int(courseImg.shape[0]/5))))
			# 	courseNo = courseNo+1
			# 	outCourses.append(courseImg)
			# print("Complete drawing")
			# # for oc in outCourses:
			# # 	cv2.imshow(practiceMove+"course", cv2.resize(oc, (int(oc.shape[1]/2), int(oc.shape[0]/2))))
			# # 	cv2.waitKey(0)
			# # 	cv2.destroyAllWindows()
		else:
			print("Failed generate course")
	else:
		print("Failed generate course")
