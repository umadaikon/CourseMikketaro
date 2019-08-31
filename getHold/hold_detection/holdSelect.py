import numpy as np
import csv
import math
import cv2
from sympy.geometry import Point, Polygon
import pylab as plt
import mysql.connector
import operator
import copy

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

#ムーブ情報
moves = list()
#0を原点としたmove情報
moveRawRange = list()
#moveRawRange = [[ムーブ名, 原点に近いp1, 原点に近いp2]]
#0を原点としたnonmove情報
nonMoveRawRange = list()
MoveHoldsRange = list()
#入力
#move = ["deadpoint","dyagonal"]
wallImgPath = "./hold_pinakuru2/IMG_9499.JPG"
moveName = ""
count = 1
wallImg = cv2.imread(wallImgPath)
baseImgSize = [4160, 3120]
#----------全ムーブデータ取得
for row in range(2):
	if count == 1:
		moveName = "deadpoint"
	elif count == 2:
		moveName = "crossmove"
	# print("Count", count)
	conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='test_db'
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

	#データベースから抽出したムーブデータ格納
	move = [[0 for column in range(6)]for row in range(5)] #rowの方のrange(5) ＝　データ数5個分のスペースを確保
	#データベースの中身
	#move = [[id[0], p_distance[0], p_angle[0], m_distance[0], m_angle[0], mainhand]

	i_max = 0
	for i, row in enumerate(cur.fetchall()):
		# print(row)
		move[i][0] = i+1     #id
		move[i][1] = row[1] * (wallImg.shape[1]/baseImgSize[1]) #p_distance
		move[i][3] = row[3] * (wallImg.shape[1]/baseImgSize[1]) #m_distance
		move[i][5] = row[5]  #mainhand

		if row[5] == 1:
			move[i][2] = row[2]
			move[i][4] = row[4]
		#movesの左主体ムーブを右主体ムーブに変換
		else:
			if row[2] <= 0: #p_angle
				move[i][2] = -180.0 - row[2]
			else:
				move[i][2] = 180.0 - row[2]
			if row[4] <= 0: #m_angle
				move[i][4] = -180.0 - row[4]
			else:
				move[i][4] = 180.0 - row[4]
			move[i][5] = 1 #mainhand
		i_max = i+1
		# print('i_max', i_max)
		moves.append([move[i][0],move[i][1],move[i][2],move[i][3],move[i][4],move[i][5]])
	#空要素の配列を削除
	move = list(filter(lambda lstx:lstx[0]!=0,move))
	# print(move)

#----------全ムーブデータ取得
	#----------ムーブ範囲算出
	#ムーブ後座標
	x = np.arange(i_max, dtype = 'float64')
	y = np.arange(i_max, dtype = 'float64')

	for i, row in enumerate(move):
		#データを座標に変換
		x[i] = math.cos(move[i][4] * math.pi/180) * move[i][3]
		y[i] = math.sin(move[i][4] * math.pi/180) * move[i][3]
	x = [x_ for y_, x_ in sorted(zip(y, x))]
	y.sort()
	p1 = np.array([0.0,0.0])
	p2 = np.array([0.0,0.0])
	p3 = np.array([0.0,0.0])
	p4 = np.array([0.0,0.0])

	Inclination = inclination(x[0], y[0], x[1], y[1])
	#データ1とデータ2のどちらが上にあるか判別しないといけない
	#データ1が上という前提
	allowRange = 40
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
	# print("p1", p1)
	# print("p2", p2)
	# print("p3", p3)
	# print("p4", p4)

	#ムーブ座標格納
	moveRawRange.append([moveName, p1, p2, p3, p4])
	# print("Move raw range", moveRawRange)
	#----------ムーブ範囲算出

	#----------nonムーブ範囲算出
	# ps = [pss for pnorm, pss in sorted(zip([np.linalg.norm(p1), np.linalg.norm(p2), np.linalg.norm(p3), np.linalg.norm(p4)], [p1, p2, p3, p4]))][:2]
	# d, b = line(ps[0][0], ps[0][1], ps[1][0], ps[1][1])
	# d0 = inclination(0, 0, ps[0][0], ps[0][1])
	# d1 = inclination(0, 0, ps[1][0], ps[1][1])
	# nonMoveRawRange.append(intersection(d, b, d0, intercept(d0, ps[0][0], ps[0][1])))
	# nonMoveRawRange.append(intersection(d, b, d1, intercept(d1, ps[1][0], ps[1][1])))
	if np.linalg.norm(p1) < np.linalg.norm(p2):
		d14, b14 = line(p1[0], p1[1], p4[0], p4[1])
		d1 = inclination(0, 0, x[0], y[0])
		d4 = inclination(0, 0, x[1], y[1])
		nonMoveRawRange.append(intersection(d14, b14, d1, intercept(d1, x[0], y[0])))
		nonMoveRawRange.append(intersection(d14, b14, d4, intercept(d4, x[1], y[1])))
	else:
		d23, b23 = line(p2[0], p2[1], p3[0], p3[1])
		d2 = inclination(0, 0, x[0], y[0])
		d3 = inclination(0, 0, x[1], y[1])
		nonMoveRawRange.append(intersection(d23, b23, d2, intercept(d2, x[0], y[0])))
		nonMoveRawRange.append(intersection(d23, b23, d3, intercept(d3, x[1], y[1])))
	#----------nonムーブ範囲算出
	count = count+1
#-----------------------------------------------------
#nonMoveRawRangeを整理
#角度（絶対値）算出
nonMoveRawAngle = [np.rad2deg(np.arccos(np.clip(np.inner(nmrr, np.array([1.0, 0.0]))/(np.linalg.norm(nmrr) * np.linalg.norm(np.array([1.0, 0.0]))), -1.0, 1.0))) for nmrr in nonMoveRawRange]
#y軸が負のものにマイナスを付与
nonMoveRawAngle = [-nmra if nmrr[1] < 0 else nmra for nmra, nmrr in zip(nonMoveRawAngle, nonMoveRawRange)]
#90度を始めとしてソート
nMRRA = zip(nonMoveRawRange, nonMoveRawAngle)
nMRRA_ = [nmrra for nmrra in nMRRA if nmrra[1]<90.0]
nMRRA__ = [nmrra for nmrra in nMRRA if nmrra[1]>=90.0]
nMRRA_.sort(key=operator.itemgetter(1))
nMRRA__.sort(key=operator.itemgetter(1))
nonMoveRawRange = [nmrr for nmrr, nmra in nMRRA_] + [nmrr for nmrr, nmra in nMRRA__]

def HLowSV(imgpass):
	hsv = cv2.cvtColor(cv2.imread(imgpass), cv2.COLOR_BGR2HSV_FULL)
	hsv[:,:,1] = hsv[:,:,1] * 0.5
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

#コース
course = list()
#生成されたコースの数をカウント
courseCount = [0, 0]
#コースに使うホールド
courseHold = list()
#既に探索したホールド
moveCheckedHold = list()
nonMoveCheckedHold = list()
#moveできるホールド
movableHold = list()
#ムーブホールドが見つかればTrue
findMove = [False, False]
# #nonMoveHoldが空ならTrue
# notFoundCourse = [False, False]
#手の入れ替え
changeHand = False
keepHand = -1

# start = [[89, 330, 0], [175.5, 352.5, 0]]
# goal = [478.5, 61.5]
#始点
start = [[1474.0, 3295.0, 0], [2200.5, 2987.5, 0]]
#終点
goal = [3332.5, 179.0]

#描画関連変数
lineSize = 10
circle1Size = 50
circle2Size = 70
fontSize = 5
fontThick = 8
lengFontPoint = circle1Size * 4
leftFontPoint = -circle1Size * 2
rightFontPoint = circle1Size


moveName = "deadpoint"
course = [[], []]
courseHold = [[], []]
moveCheckedHold = [[], []]
nonMoveCheckedHold = [[], []]
movableHold = [[], []]
stackHold = [[start[0]], [start[1]]]
#stackHold = [左手[ホールドx座標, ホールドy座標, [移動前のホールド座標], if ムーブホールド], 右手[]]
hand = 1
if hand == 0:
	B = 255
	point = -10
else:
	B = 0
	point = 0
holdPoint = list()
holdRect = list()
#ホールド座標を取得
with open("./pi_9499.csv", 'r') as fr:
	reader = csv.reader(fr)
	header = next(reader)
	for rd in reader:
		holdPoint.append([float(rd[0]), float(rd[1])])
		holdRect.append([float(rd[2]), float(rd[3]), float(rd[4]), float(rd[5]), float(rd[6]), float(rd[7]), float(rd[8]), float(rd[9])])

	# print("Holdpoint", holdPoint)
	# print("Holdrect", holdRect)
	# for i, hp in enumerate(holdPoint):
	# 	wallImg = cv2.circle(wallImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 40, 255), -1)
	# for i, hr in enumerate(holdRect):
	# 	wallImg = cv2.rectangle(wallImg, (int(hr[4]), int(hr[5])), (int(hr[0]), int(hr[1])), (255, 0, 0), lineSize)
	# cv2.imwrite("Hold.png", wallImg)

#練習したいムーブのムーブ情報を取り出す
for mRR in moveRawRange:
	if mRR[0] in moveName:
		moveData = mRR
print("Move data", moveData)

#-----------コース生成開始
print("Start generate course")
while(not stackHold[0] == [] and not stackHold[1] == []):
	if not keepHand == -1:
		hand = keepHand
	if hand == 0:
		B = 255
		point = -7
	else:
		B = 0
		point = 2
	# #値確認
	# print()
	# print("Hand", hand)
	# print("keep", keepHand)
	# print("Find move", findMove[hand])
	# print("Stack", stackHold[hand])
	courseHold[hand].append(stackHold[hand][-1][:3])
	P = list([stackHold[hand][-1][:2]])
	# print("Pop", stackHold[hand][-1])
	# print("Course hold", courseHold[hand])
	# print("findMove", findMove)
	del stackHold[hand][-1]
	if keepHand == -1:
		if (courseHold[hand][-1] == [goal[0], goal[1], 0] or courseHold[hand][-1] == [goal[0], goal[1], 1]) and findMove[hand] == True:
			# #値確認
			# print()
			# print("Course No", courseCount[hand] , "Hand", hand, courseHold[hand])
			# print("0", findMove)

			course[hand].append(copy.deepcopy(courseHold[hand]))

			try:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 0])
			except ValueError:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 1])
			# print("Del", courseHold[hand][backHoldIndex+1:])
			del courseHold[hand][backHoldIndex+1:]
			# del checkedHold[hand][-1]
			moveCheckedHold[hand] = []
			nonMoveCheckedHold[hand] = []
			findMove[hand] = False
			courseCount[hand] = courseCount[hand]+1
			if hand == 0:
				keepHand = 1
			else:
				keepHand = 0
			continue
	else:
		if courseHold[hand][-1] == [goal[0], goal[1], 0] or courseHold[hand][-1] == [goal[0], goal[1], 1]:
			# #値確認
			# print()
			# print("Course No", courseCount[hand] , "Hand", hand, courseHold[hand])
			# print("1", findMove)

			course[hand].append(copy.deepcopy(courseHold[hand]))

			try:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 0])
			except ValueError:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 1])
			# print("Del", courseHold[hand][backHoldIndex+1:])
			del courseHold[hand][backHoldIndex+1:]
			# del checkedHold[hand][-1]
			moveCheckedHold[hand] = []
			nonMoveCheckedHold[hand] = []
			findMove[hand] = False
			courseCount[hand] = courseCount[hand]+1
			keepHand = -1
			if hand == 0:
				hand = 1
			else:
				hand = 0
			continue

	#-----------ムーブ範囲確認
	if not findMove[hand]:
		mP = list()
		if hand == 0:
			mP.append([P[0][0]-moveData[1][0], P[0][1]+moveData[1][1]])
			mP.append([P[0][0]-moveData[2][0], P[0][1]+moveData[2][1]])
			mP.append([P[0][0]-moveData[3][0], P[0][1]+moveData[3][1]])
			mP.append([P[0][0]-moveData[4][0], P[0][1]+moveData[4][1]])
		else:
			mP.append([P[0][0]+moveData[1][0], P[0][1]+moveData[1][1]])
			mP.append([P[0][0]+moveData[2][0], P[0][1]+moveData[2][1]])
			mP.append([P[0][0]+moveData[3][0], P[0][1]+moveData[3][1]])
			mP.append([P[0][0]+moveData[4][0], P[0][1]+moveData[4][1]])

		MoveHoldRange = Polygon(*mP)
		MoveHoldsRange.append(MoveHoldRange)
		cur.close()
		# conn.commit()
		conn.close()

		# #値確認
		# print("Moveholdrange", *mP)

		MoveHold = [hp for hp in holdPoint if MoveHoldRange.encloses_point(hp)]

		# #値確認
		# print("Movehold", MoveHold)
		# print("Checked(move)", moveCheckedHold[hand])

		for i, mh in reversed(list(enumerate(MoveHold))):
			for checked in moveCheckedHold[hand]:
				if mh == checked:
					del MoveHold[i]
					break

		# print("Checked movehold", MoveHold)

		if not MoveHold == []:
			MoveHold = AngleSort(P[0], goal, MoveHold)
			movableHold[hand].extend(MoveHold)
			changeHand = True


		# #描画
		# moveImg = HLowSV(wallImgPath)
		# #ムーブ範囲の描画
		# for i in range(len(mP)-1):
		# 	moveImg = cv2.line(moveImg, (int(mP[i][0]), int(mP[i][1])), (int(mP[i+1][0]), int(mP[i+1][1])), (0, 170, 0), lineSize)
		# moveImg = cv2.line(moveImg, (int(mP[i+1][0]), int(mP[i+1][1])), (int(mP[0][0]), int(mP[0][1])), (0, 170, 0), lineSize)
		# #ムーブホールドの描画
		# for i, mh in enumerate(MoveHold):
		# 	moveImg = cv2.circle(moveImg, (int(mh[0]), int(mh[1])), circle2Size, (0, 255, 255), -1)
		# #ホールド座標の描画
		# for i, hp in enumerate(holdPoint):
		# 	moveImg = cv2.circle(moveImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 60, 255), -1)
		#
		# #始点の描画
		# moveImg = cv2.circle(moveImg, (int(P[0][0]), int(P[0][1])), circle2Size, (0, 255, 0), -1)
		# for i, ch in enumerate(courseHold[hand]):
		# 	moveImg = cv2.putText(moveImg, str(i), (int(ch[0]+point), int(ch[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, (B, 0, 255-B))
		# cv2.imshow("MoveHoldRange", moveImg)
	#-----------ムーブ範囲確認

	#-----------nonムーブ範囲算出
	for nmrr in nonMoveRawRange:
		if hand == 0:
			P.append([P[0][0]-nmrr[0], P[0][1]+nmrr[1]])
		else:
			P.append([P[0][0]+nmrr[0], P[0][1]+nmrr[1]])

	#点の範囲
	nonMoveHoldRange = Polygon(*P)
	# #値確認
	# print("Non moveholdrange:", *P)

	#----------nonムーブ範囲確認
	#範囲内にあるホールドを取得
	nonMoveHold = [hp for hp in holdPoint if nonMoveHoldRange.encloses_point(hp)]

	# #値確認
	# print("Nonmovehold", nonMoveHold)
	# print("Checked(nonMove)", nonMoveCheckedHold[hand])

	for i, nmh in reversed(list(enumerate(nonMoveHold))):
		for checked in nonMoveCheckedHold[hand]:
			if nmh == checked:
				del nonMoveHold[i]
				break
	# #値確認
	# print("Checked nonmoveHold", nonMoveHold)
	#----------nonムーブ範囲確認

	if not nonMoveHold == []:
		nonMoveHold = AngleSort(P[0], goal, nonMoveHold)
		changeHand = True
	else:
		# print(stackHold)
		if stackHold[hand] == []:
			if keepHand == 0:
				keepHand = 1
			else:
				keepHand = 0
		else:
			try:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 0])
			except ValueError:
				backHoldIndex = courseHold[hand].index([stackHold[hand][-1][3][0], stackHold[hand][-1][3][1], 1])
			# #値確認
			# print("Del", courseHold[hand][backHoldIndex+1:])
			# print("Movable hold", movableHold[hand])
			for i, mbh in reversed(list(enumerate(movableHold[hand]))):
				for chd in courseHold[hand][backHoldIndex+1:]:
					if mbh == chd[:2]:
						del movableHold[hand][i]
						break
			# print("Movable hold deled", movableHold[hand])
			if movableHold[hand] == []:
				findMove[hand] = False
			del courseHold[hand][backHoldIndex+1:]
		# notFoundCourse[hand] = True

	# #描画
	# nonMoveImg = HLowSV(wallImgPath)
	# #移動範囲の描画
	# for i in range(len(P)-1):
	# 	nonMoveImg = cv2.line(nonMoveImg, (int(P[i][0]), int(P[i][1])), (int(P[i+1][0]), int(P[i+1][1])), (0, 170, 0), lineSize)
	# nonMoveImg = cv2.line(nonMoveImg, (int(P[i+1][0]), int(P[i+1][1])), (int(P[0][0]), int(P[0][1])), (0, 170, 0), lineSize)
	# #移動可能ホールドの描画
	# for i, nmh in enumerate(nonMoveHold):
	# 	# print("Non move hold", nmh)
	# 	nonMoveImg = cv2.circle(nonMoveImg, (int(nmh[0]), int(nmh[1])), circle2Size, (0, 255, 255), -1)
	# # ホールド座標の描画
	# for i, hp in enumerate(holdPoint):
	# 	nonMoveImg = cv2.circle(nonMoveImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 60, 255), -1)
	# #始点の描画
	# nonMoveImg = cv2.circle(nonMoveImg, (int(P[0][0]), int(P[0][1])), circle2Size, (0, 255, 0), -1)
	# for i, ch in enumerate(courseHold[hand]):
	# 	nonMoveImg = cv2.putText(nonMoveImg, str(i), (int(ch[0]+point), int(ch[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, (B, 0, 255-B))
	# cv2.imshow("nonMoveHoldRange", nonMoveImg)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	#追加
	if not nonMoveHold == []:
		for nmh in nonMoveHold:
			stackHold[hand].append([nmh[0], nmh[1], 0, P[0]])
		nonMoveCheckedHold[hand].extend(nonMoveHold)
		# #値確認
		# print("Extend non", nonMoveHold)
	if not MoveHold == []:
		for mh in MoveHold:
			stackHold[hand].append([mh[0], mh[1], 1, P[0]])
		moveCheckedHold[hand].extend(MoveHold)
		findMove[hand] = True
		# #値確認
		# print("Extend", MoveHold)
		MoveHold.clear()
	#追加------

	# #画像出力による範囲チェック
	# x = input("writing?(y/n):")
	# if x == "y":
	# 	cv2.imwrite("move.png", moveImg)
	# 	cv2.imwrite("nonMove.png", nonMoveImg)

	if changeHand == True:
		if hand == 0:
			hand = 1
		else:
			hand = 0
		changeHand = False
#-----------コース生成終了
print()
print("Course data", course)
if not course[0] == [] and not course[1] == []:
	print("Complete generate course")
	#コースホールドの描画
	for i, (l, r) in enumerate(zip(course[0], course[1])):
		#値確認
		print("Left", l)
		print("Right", r)
		courseImg = HLowSV(wallImgPath)
		#ホールド座標の描画
		# for j, hp in enumerate(holdPoint):
		# 	courseImg = cv2.circle(courseImg, (int(hp[0]), int(hp[1])), 5, (255, 200, 255), -1)
		for No, lhold in enumerate(l):
			if not No == 0:
				courseImg = cv2.line(courseImg, (int(l[No-1][0]), int(l[No-1][1])), (int(lhold[0]), int(lhold[1])), (255, 0, 0), lineSize)
		for No, rhold in enumerate(r):
			if not No == 0:
				courseImg = cv2.line(courseImg, (int(r[No-1][0]), int(r[No-1][1])), (int(rhold[0]), int(rhold[1])), (0, 0, 255), lineSize)

		for No, lhold in enumerate(l):
			if lhold[2] == 1:
				courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle1Size, (0, 170, 0), -1)
			else:
				courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle2Size, (0, 0, 0), -1)
			holdIndex = holdPoint.index(lhold[:2])
			# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
			courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle1Size, (255, 0, 0), -1)
		for No, rhold in enumerate(r):
			if rhold[2] == 1:
				courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 170, 0), -1)
			else:
				courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 0, 0), -1)
				holdIndex = holdPoint.index(rhold[:2])
			# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
			courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle1Size, (0, 0, 255), -1)

		for No, lhold in enumerate(l):
			courseImg = cv2.putText(courseImg, str(No), (int(lhold[0]+leftFontPoint), int(lhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 0), fontThick)
		for No, rhold in enumerate(r):
			courseImg = cv2.putText(courseImg, str(No), (int(rhold[0]+rightFontPoint), int(rhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThick)
		# 始点、終点の描画
		# courseImg = cv2.circle(courseImg, (int(start[0][0]), int(start[0][1])), 3, (0, 255, 255), -1)
		# courseImg = cv2.circle(courseImg, (int(start[1][0]), int(start[1][1])), 3, (0, 255, 255), -1)
		# courseImg = cv2.circle(courseImg, (int(goal[0]), int(goal[1])), 3, (0, 255, 255), -1)
		cv2.imshow("course"+str(i), courseImg)
		cv2.imwrite("./course"+str(i)+".png", courseImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

else:
	print("Failed generate course")
