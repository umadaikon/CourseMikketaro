# -*- coding: utf-8 -*-

import math
import pylab as plt
import csv
import numpy as np
import mysql.connector
import numpy as np

filename = "./resource/crossmove_1.csv"
#csv_read
csv_file = open(
	filename,
	"r",
	encoding="ms932",
	errors="",
	newline=""
)
#リスト形式
f = csv.reader(
	csv_file,
	delimiter=",",
	doublequote=True,
	lineterminator="\r\n",
	quotechar='"',
	skipinitialspace=True
)
header = next(f)
#print(header) #csv内の要素確認
#csvの中身
#cogx, cogy, p_beforex, p_beforey, before_afterx, before_aftery, mainhand, preframe, postframe
p_beforex = 0.0
p_beforey = 0.0
b_ax = 0.0
b_ay = 0.0
mainhand = 1 #デフォルトは右手がムーブの主体

for row in f:
	#rowはList
	#row[0]で必要な項目を取得することができる
	#print(row)
	p_beforex = row[2]
	p_beforey = row[3]
	b_ax = row[4]
	b_ay = row[5]
	mainhand = row[6]

#左手が主体の手の場合はx座標を反転
if mainhand == "0":
	p_beforex = -float(p_beforex)
	b_ax = -float(b_ax)

print("mainhand", mainhand)
#X[start:end:step]
# print (rightx[0], righty[0])

stay_p = np.array([float(p_beforex), float(p_beforey)], dtype = object)
before_p = np.array([0.0, 0.0], dtype = object)
after_p = np.array([float(b_ax), float(b_ay)], dtype = object)

radian = np.arange(2.0)
radian[0] = math.atan2(stay_p[1]-before_p[1], stay_p[0]-before_p[0])
radian[1] = math.atan2(after_p[1]-before_p[1], after_p[0]-before_p[0])

#print ("deg", math.degrees(result1))
angle = np.arange(2.0)
angle[0] = math.degrees(radian[0])
angle[1] = math.degrees(radian[1])
'''
#正の角度で出力
if deg < 0:
	deg = -deg
'''
# print ("angle", angle)

distance = np.arange(2.0)
distance[0] = np.linalg.norm(stay_p - before_p)
distance[1] = np.linalg.norm(after_p - before_p)

#distance[distance < 0] = -distance
for i, row in enumerate(distance):
	if distance[i] < 0:
		distance[i] = -row

print ("distance", distance)

# plt.plot([rightx[0], rightx[1]], [righty[0], righty[1]], 'k-')
# plt.show()
csv_file.close()


#database
conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='bouldering_db'
)
cur = conn.cursor()

#入力
#move = ["deadpoint","dyagonal"]

move = "crossmove"
id = 2

value = "values(" + str(id) + ", " + str(p_beforex) + ", " + str(p_beforey) + ", " + str(b_ax) + ", " + str(b_ay) + ", " + str(mainhand) + ");"
i_command = "insert into " + move + "(id, p_distance, p_angle, m_distance, m_angle, mainhand) " + value
s_command = "select * from " + move + ";"

#print (value)
#print (i_command)
#print (s_command)
cur.execute(i_command)
cur.execute(s_command)

#データベースから抽出したムーブデータ格納
move = [[0 for column in range(6)]for row in range(5)] #rowの方のrange(5) ＝　データ数5個分のスペースを確保
#データベースの中身
#move = [[id[0], p_distance[0], p_angle[0], m_distance[0], m_angle[0], mainhand]

i_max = 0
for i, row in enumerate(cur.fetchall()):
	#print(row[1],row[2])
	move[i][0] = i+1     #id
	move[i][1] = row[1]  #p_distance
	move[i][2] = row[2]  #p_angle
	move[i][3] = row[3]  #m_distance
	move[i][4] = row[4]  #m_angle
	move[i][5] = row[5]  #mainhand
	i_max = i+1

print(i_max)
x = np.arange(i_max, dtype = 'float64')
y = np.arange(i_max, dtype = 'float64')

#print(move)


#空要素の配列を削除
move = list(filter(lambda lstx:lstx[0]!=0,move))
print(move)


#傾き
def inclination(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)
#切片
def intercept(d,x,y):
    return y-d*x
#2直線の交点x座標
def intersection(d1,b1,d2,b2):
    return (b2-b1)/(d1-d2)

#ムーブ前座標
start = [[0 for column in range(2)]for row in range(i_max)]
#ムーブ後座標
finish = np.array([[0 for column in range(2)]for row in range(i_max)], dtype = object)

# print(start)
# print(finish)
#mainhandのみ取得
for i, row in enumerate(range(i_max)):
	#右手
	if(move[i][5] == 1):
		#データを座標に変換
		x[i] = math.cos(move[i][4] * math.pi/180) * move[i][3]
		y[i] = math.sin(move[i][4] * math.pi/180) * move[i][3]
		#ムーブ前を原点に設定したムーブ後の座標を算出
		finish[i] = [(start[i][0]+x[i]), (start[i][1]+y[i])]
		# print(x)
		continue

	#左手
	elif(move[i][5] == 0):
		#データを座標に変換
		x[i] = math.cos(move[i][2] * math.pi/180) * move[i][1]
		y[i] = math.sin(move[i][2] * math.pi/180) * move[i][1]
		#ムーブ前を原点に設定したムーブ後の座標を算出
		finish[i] = [(start[i][0]+x[i]), (start[i][1]+y[i])]
		continue

d = inclination((start[0][0]+x[0]), (start[0][1]+y[0]), (start[1][0]+x[1]), (start[1][1]+y[1]))

# print(x)
# print(y)
# print(finish)

#四角形の面積計算準備
#平行移動
Inclination = inclination(finish[0][0], finish[0][1], finish[1][0], finish[1][1])
p1 = np.array([0.0,0.0])
p2 = np.array([0.0,0.0])
p3 = np.array([0.0,0.0])
p4 = np.array([0.0,0.0])

#データ1とデータ2のどちらが上にあるか判別しないといけない
# print(type(p1))
#データ1が上という前提
if(Inclination > 0):
	p1 = np.array([finish[0][0]+10, finish[0][1]+10])
	p2 = np.array([finish[0][0]-10, finish[0][1]-10])
	p3 = np.array([finish[1][0]+10, finish[1][1]+10])
	p4 = np.array([finish[1][0]-10, finish[1][1]-10])
elif(Inclination < 0):
	p1 = np.array([finish[0][0]+10, finish[0][1]-10])
	p2 = np.array([finish[0][0]-10, finish[0][1]+10])
	p3 = np.array([finish[1][0]-10, finish[1][1]+10])
	p4 = np.array([finish[1][0]+10, finish[1][1]-10])
elif(Inclination == 0):
	p1 = np.array([finish[0][0], finish[0][1]+10])
	p2 = np.array([finish[0][0], finish[0][1]-10])
	p3 = np.array([finish[1][0], finish[1][1]+10])
	p4 = np.array([finish[1][0], finish[1][1]-10])
else:
	p1 = np.array([finish[0][0]+10, finish[0][1]])
	p2 = np.array([finish[0][0]-10, finish[0][1]])
	p3 = np.array([finish[1][0]+10, finish[1][1]])
	p4 = np.array([finish[1][0]-10, finish[1][1]])

# print(type(p1))
print(p1)
print(p2)
print(p3)
print(p4)

#四角形の面積計算
rectangle = np.arange(2.0)
rectangle[0] = np.linalg.norm(p1 - p2)
rectangle[1] = np.linalg.norm(p2 - p3)

#distance[distance < 0] = -distance
for i, row in enumerate(rectangle):
	if rectangle[i] < 0:
		rectangle[i] = -row

print ("rectangle", rectangle)
'''
area = 0.0
area = rectangle[0]*rectangle[1]
print(area)
'''

cur.close()
conn.commit()
conn.close()
