# -*- coding: utf-8 -*-
#python csv_db_access.py csvファイル名
import mysql.connector
import math
import pylab as plt
import csv
import numpy as np
import sys

#mysql
conn = mysql.connector.connect(
	user='root',
	password='',
	host='localhost',
	database='test3_db'
)
cur = conn.cursor(buffered=True)

filename = sys.argv[1]

#csv_read
csv_file = open(
	"./resource/"+filename,
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
#cogx, cogy, p_beforex, p_beforey, before_afterx, before_aftery, mainhand
'''
rightx = 0.0
righty = 0.0
leftx = 0.0
lefty = 0.0
'''
p_beforex = 0.0
p_beforey = 0.0
b_ax = 0.0
b_ay = 0.0
mainhand = 1 #デフォルトは右手がムーブの主体

for row in f:
	#rowはList
	#row[0]で必要な項目を取得することができる
	#print(row)
	'''
	leftx=row[2]
	lefty=row[3]
	rightx=row[4]
	righty=row[5]
	'''
	p_beforex = row[2]
	p_beforey = row[3]
	b_ax = row[4]
	b_ay = row[5]
	mainhand = row[6]

print("mainhand", mainhand)
#X[start:end:step]
# print (rightx[0], righty[0])

'''
#csv_move_da
before_l = np.array([0.0, 0.0])
after_l = np.array([leftx, lefty])
before_r = np.array([0.0, 0.0])
after_r = np.array([rightx, righty])
'''
stay_p = np.array([float(p_beforex), float(p_beforey)], dtype = object)
before_p = np.array([0.0, 0.0], dtype = object)
after_p = np.array([float(b_ax), float(b_ay)], dtype = object)

'''
#result = math.atan2(ay-by, ax-bx)
radian = np.arange(2.0)
radian[0] = math.atan2(lefty-before_l[1], leftx-before_l[0])
radian[1] = math.atan2(righty-before_r[1], rightx-before_r[0])
#print ("rad", radian)
'''
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

'''
#distance = (ay-by)/(ax-bx)
distance = np.arange(2.0)
distance[0] = np.linalg.norm(after_l - before_l)
distance[1] = np.linalg.norm(after_r - before_r)
'''
distance = np.arange(2.0)
distance[0] = np.linalg.norm(stay_p - before_p)
distance[1] = np.linalg.norm(after_p - before_p)

#distance[distance < 0] = -distance
for i, row in enumerate(distance):
	if distance[i] < 0:
		distance[i] = -row

# print ("distance", distance)

# plt.plot([rightx[0], rightx[1]], [righty[0], righty[1]], 'k-')
# plt.show()

csv_file.close()

#mysql
#move = ["deadpoint","dyagonal"]
id = filename.split("_")[1].split(".")[0]
move = filename.split("_")[0]

# print(cur.fetchall())

#決めうち確認
#distance = 5
#angle = 45

value = "values(" + str(id) + ", " + str(distance[0]) + ", " + str(angle[0]) + ", " + str(distance[1]) + ", " + str(angle[1]) + ", " + str(mainhand) + ");"
i_command = "insert into " + move + "(id, p_distance, p_angle, m_distance, m_angle, mainhand) " + value
s_command = "select * from " + move + ";"

print (value)
print (i_command)
print (s_command)
cur.execute(i_command)
cur.execute(s_command)
'''
for row in cur.fetchall():
	print(row[0],row[1],row[2],row[3],row[4])
'''

cur.close()
conn.commit()
conn.close()
