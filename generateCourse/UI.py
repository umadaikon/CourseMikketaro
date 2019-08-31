import cv2
import csv

#-------ここから前準備
#彩度を下げるついでに画像読み込み、サイズ変更
def HLowSV(imgpass):
	hsv = cv2.cvtColor(cv2.resize(cv2.imread(imgpass), (600, 900)), cv2.COLOR_BGR2HSV_FULL)
	hsv[:,:,1] = hsv[:,:,1] * 0.5
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

#描画関連変数
lineSize = 4
circle1Size = 10
circle2Size = 15
fontSize = 1
fontThick = 1
lengFontPoint = circle1Size * 4
leftFontPoint = -circle1Size * 2
rightFontPoint = circle1Size
B = 255
point = -10

#ホールド座標
holdPoint = list()
#ホールド四角座標
holdRect = list()

#ウォール画像のパス
wallImgPath = "./IMG_9499.JPG"
wallImg = cv2.imread(wallImgPath)
wallImg = cv2.resize(wallImg, (600, 900))
with open("./pi_9499.csv", 'r') as fr:
	reader = csv.reader(fr)
	header = next(reader)
	imgHeight = wallImg.shape[0]
	for rd in reader:
		# holdPoint.append([float(rd[1]), imgHeight-float(rd[0])])
		# holdRect.append([float(rd[3]), imgHeight-float(rd[2]), float(rd[5]), imgHeight-float(rd[4]), float(rd[7]), imgHeight-float(rd[6]), float(rd[9]), imgHeight-float(rd[8])])
		holdPoint.append([float(rd[0]), float(rd[1])])
		holdRect.append([float(rd[2]), float(rd[3]), float(rd[4]), float(rd[5]), float(rd[6]), float(rd[7]), float(rd[8]), float(rd[9])])
	# #値確認
	# print("Holdpoint", len(holdPoint))
	# print("Holdrect", holdRect)
	# for i, hp in enumerate(holdPoint):
	# 	wallImg = cv2.circle(wallImg, (int(hp[0]), int(hp[1])), circle1Size, (255, 40, 255), -1)
	#全てのホールドの描画
	# for i, hr in enumerate(holdRect):
	# 	wallImg = cv2.rectangle(wallImg, (int(hr[4]), int(hr[5])), (int(hr[0]), int(hr[1])), (255, 60, 255), lineSize)
	# cv2.imwrite("Hold.png", wallImg)
	# for No, hp in enumerate(holdPoint):
	# 	wallImg = cv2.putText(wallImg, str(No), (int(hp[0]), int(hp[1])), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThick)
	# wallImg = cv2.resize(wallImg, (int(wallImg.shape[1]), int(wallImg.shape[0])))
# cv2.imshow("wall", wallImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-------ここまで前準備


#コースホールド
course = [[[[409.0, 703.5, 0],[345.0, 520.5, 1],[354.0, 438.5, 0],[400.0, 282.5, 0],[421.5, 220.5, 0]]],[[[493.0, 697.5, 0],[460.5, 546.0, 0],[415.5, 442.5, 0],[476.5, 335.0, 0],[421.5, 220.5, 0]]]]

print("Course data", course)
if not course[0] == [] and not course[1] == []:
	print("Complete generate course")
	for i, (l, r) in enumerate(zip(course[0], course[1])):
	#l:左手のコースホールド座標
	#r:右手のコースホールド座標
		# #値確認
		# print("Left", l)
		# print("Right", r)
		courseImg = HLowSV(wallImgPath)
		#全ホールドの描画
		# for j, hp in enumerate(holdPoint):
		# 	courseImg = cv2.circle(courseImg, (int(hp[0]), int(hp[1])), 5, (255, 200, 255), -1)

		#コースをつなぐ線の描画
		for No, lhold in enumerate(l):
			if not No == 0:
				courseImg = cv2.line(courseImg, (int(l[No-1][0]), int(l[No-1][1])), (int(lhold[0]), int(lhold[1])), (255, 0, 0), lineSize)
		for No, rhold in enumerate(r):
			if not No == 0:
				courseImg = cv2.line(courseImg, (int(r[No-1][0]), int(r[No-1][1])), (int(rhold[0]), int(rhold[1])), (0, 0, 255), lineSize)

		#ホールドの描画
		for No, lhold in enumerate(l):
			if lhold[2] == 1:
				#ムーヴ可能ホールド時の強調表示
				courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle2Size, (0, 170, 0), -1)
			else:
				#そのほかのホールドの強調表示
				courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle2Size, (0, 0, 0), -1)
			#右手のホールドの描画
			courseImg = cv2.circle(courseImg, (int(lhold[0]), int(lhold[1])), circle1Size, (255, 0, 0), -1)
			# ホールドを四角で囲む
			# holdIndex = holdPoint.index(lhold[:2])
			# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (255, 0, 0), lineSize)
		for No, rhold in enumerate(r):
			if rhold[2] == 1:
				#ムーヴ可能ホールド時の強調表示
				courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 170, 0), -1)
			else:
				#そのほかのホールドの強調表示
				courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle2Size, (0, 0, 0), -1)
			#左手のホールドの描画
			courseImg = cv2.circle(courseImg, (int(rhold[0]), int(rhold[1])), circle1Size, (0, 0, 255), -1)
			#ホールドを四角で囲む
			# holdIndex = holdPoint.index(rhold[:2])
			# courseImg = cv2.rectangle(courseImg, (int(holdRect[holdIndex][4]), int(holdRect[holdIndex][5])), (int(holdRect[holdIndex][0]), int(holdRect[holdIndex][1])), (0, 0, 255), lineSize)
		#順番の描画
		for No, lhold in enumerate(l):
			courseImg = cv2.putText(courseImg, str(No), (int(lhold[0]+leftFontPoint), int(lhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 0), fontThick)
		for No, rhold in enumerate(r):
			courseImg = cv2.putText(courseImg, str(No), (int(rhold[0]+rightFontPoint), int(rhold[1])+lengFontPoint), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), fontThick)
		# 始点、終点の描画
		# courseImg = cv2.circle(courseImg, (int(start[0][0]), int(start[0][1])), 3, (0, 255, 255), -1)
		# courseImg = cv2.circle(courseImg, (int(start[1][0]), int(start[1][1])), 3, (0, 255, 255), -1)
		# courseImg = cv2.circle(courseImg, (int(goal[0]), int(goal[1])), 3, (0, 255, 255), -1)
		cv2.imwrite("./course"+str(i)+".png", courseImg)
		cv2.imshow("course"+str(i), cv2.resize(courseImg, (int(courseImg.shape[1]), int(courseImg.shape[0]))))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
