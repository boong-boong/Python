import cv2
from utils import image_show

# 이미지 경로
image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
img180 = cv2.rotate(image, cv2.ROTATE_180)
img270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# # 회전
# cv2.imshow("orginal image", image)
# cv2.imshow("rotate_90", img90)
# cv2.imshow("rotate_180", img180)
# cv2.imshow("rotate_270", img270)
# cv2.waitKey(0)

# 1 좌우 반전, 0 상하 반전
dst_temp01 = cv2.flip(image, 0)
dst_temp02 = cv2.flip(image, 1)

cv2.imshow("dst_temp01", dst_temp01)
cv2.imshow("dst_temp02", dst_temp02)
cv2.waitKey(0)
