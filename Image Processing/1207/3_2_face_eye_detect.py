import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

face_img = cv2.imread('face.png')  # 파일명 숫자로 시작할 경우 오류 날 수 있음

face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(face_gray, 1.1, 3)

for (x, y, w, h) in faces:  # 파일명 숫자로 시작할 경우 오류 날 수 있음
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # 하나의 결과만 나옴
    # 관심 영역 밖에도 눈으로 인식하는 것이 존재, 인식된 얼굴에만 범위 제한
    roi_img = face_img[y: y+h, x: x+w]
    roi_gray = face_gray[y: y + h, x: x + w]

    # cv2.imshow('roi', roi_img)
    # cv2.waitKey(0)
    '''
    여러개 일때
    roi_img.append(face_img[y: y+h, x: x+w])
    roi_gray.append(face_gray[y: y + h, x: x + w])
    '''

# cv2.imshow('face box', face_img)
# cv2.waitKey(0)

eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

# 눈 위치 저장
idx = 0
for (x, y, w, h) in eyes:
    if idx == 0:
        eye_1 = (x, y, w, h)
    if idx == 1:
        eye_2 = (x, y, w, h)
    cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    idx += 1

# cv2.imshow('eye box', face_img)
# cv2.waitKey(0)

# x가 더 작은 쪽이 왼쪽 눈
if eye_1[0] < eye_2[0]:
    left_eye = eye_1
    right_eye = eye_2
else:
    left_eye = eye_2
    right_eye = eye_1

# 눈의 중앙값 계산
left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))

# cv2.circle(roi_img, left_eye_center, 5, (0, 200, 200), -1)
# cv2.circle(roi_img, right_eye_center, 5, (0, 200, 200), -1)
# cv2.imshow('median eye', face_img)
# cv2.waitKey(0)

# 각도 계산
delta_x = right_eye_center[0] - left_eye_center[0]
delta_y = right_eye_center[1] - left_eye_center[1]
angle = np.arctan(delta_y/delta_x)
angle = (angle * 180) / np.pi

# 이미지를 각도 만큼 회전
h, w = face_img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(face_img, M, (w, h))
cv2.imshow('rotated img', rotated)
cv2.waitKey(0)
