import cv2
import numpy as np

img = cv2.imread("./IMG_1767.jpg")
print(img.shape)
img = cv2.circle(img, (1500, 1500), 1000, (0,0,255), -1)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("circle.jpg", img)
