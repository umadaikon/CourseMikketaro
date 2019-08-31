import cv2

bgr = cv2.imread("resource/test0213.png")

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

lab_planes = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))

lab_planes[0] = clahe.apply(lab_planes[0])

lab = cv2.merge(lab_planes)

bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow("CLAHE", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("CLAHE.png", bgr)
