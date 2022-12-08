import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

img = cv2.imread('./aespa.jpg')
face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(face_gray, 1.2, 3)

face_crop = []
for (x, y, w, h) in faces:
    face_crop.append(img[y:y+h, x:x+w])

face_crop[1], face_crop[3] = face_crop[3], face_crop[1]

# ex-01
img_rectangle = np.ones((400, 400), dtype='uint8')
cv2.rectangle(img_rectangle, (50, 50), (300, 300), (255, 255, 255), -1)
cv2.imshow("img_rectangle", img_rectangle)
cv2.waitKey(0)

# ex-02
img_circle = np.ones((400, 400), dtype='uint8')
cv2.circle(img_circle, (300, 300), 70, (255, 255, 255), -1)
cv2.imshow("img_circle", img_circle)
cv2.waitKey(0)

# ex-03
bitwiseAnd = cv2.bitwise_and(img_rectangle, img_circle)
cv2.imshow("image bitwiseAnd", bitwiseAnd)
cv2.waitKey(0)

bitwiseOr = cv2.bitwise_or(img_rectangle, img_circle)
cv2.imshow("image bitwiseOr", bitwiseOr)
cv2.waitKey(0)

bitwiseXor = cv2.bitwise_xor(img_rectangle, img_circle)
cv2.imshow("Xor", bitwiseXor)
cv2.waitKey(0)

rec_not = cv2.bitwise_not(img_rectangle)
cv2.imshow("rectangle not ", rec_not)
cv2.waitKey(0)

circle_not = cv2.bitwise_not(img_circle)
cv2.imshow("circle not ", circle_not)
cv2.waitKey(0)

# ex-04 마스킹 과제는 흰색대신 이미지를 넣어주시면 됩니다. (원하는 이미지 혹은 얼굴이미지)
mask = np.zeros((683, 1024, 3), dtype='uint8')
cv2.rectangle(mask, (60, 50), (280, 280), (255, 255, 255), -1)
cv2.rectangle(mask, (420, 50), (550, 230), (255, 255, 255), -1)
cv2.rectangle(mask, (750, 50), (920, 280), (255, 255, 255), -1)

x_offset = [60, 420, 750]
y_offset = [50, 50, 50]

x_end = [280, 550, 920]
y_end = [280, 230, 280]

for i in range(3):
    face_crop[i] = cv2.resize(face_crop[i], (x_end[i] - x_offset[i], y_end[i] - y_offset[i]))
    mask[y_offset[i]:y_offset[i]+face_crop[i].shape[0], x_offset[i]:x_offset[i]+face_crop[i].shape[1]] = face_crop[i]

cv2.imshow("...", mask)
cv2.waitKey(0)
