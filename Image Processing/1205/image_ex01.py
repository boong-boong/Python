import cv2

img_path = './cat.jpg'
img = cv2.imread(img_path)

h, w , _ = img.shape  # 높이, 넓이, 채널

print('이미지 타임 : ', type(img))
print('이미지 크기 : ', img.shape)
print(f'이미지 높이 {h}, 이미지 넓이 {w}')

'''
이미지 타임 :  <class 'numpy.ndarray'>
이미지 크기 :  (399, 600, 3) (H, W, C)
'''

cv2.imshow('image show', img)
# cv2.waitKey(0)  # 대기를 안시키면 바로 꺼짐