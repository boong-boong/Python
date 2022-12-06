# 기본적인 이미지 처리 기술을 이용한 이미지 선명화 -1
import cv2
import numpy as np

img = cv2.imread('./car.jpg', 0)
print(img.shape)
img_resize = cv2.resize(img, (320, 240))

blurred_1 = np.hstack([
    cv2.blur(img_resize, (3, 3)),
    cv2.blur(img_resize, (5, 5)),
    cv2.blur(img_resize, (9, 9))
])

cv2.imshow("show", blurred_1)
cv2.waitKey(0)
