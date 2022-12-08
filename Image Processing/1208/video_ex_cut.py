import os

import cv2

video_file_path = './video01.mp4'

# 동영상 캡쳐 객체 생성
cap = cv2.VideoCapture(video_file_path)

fps = cap.get(cv2.CAP_PROP_FPS)
print('fps >> ', fps)
count = 0
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            if int(cap.get(1)) % fps == 0:  # fps 값을 사용하여 1초마다 추출
                os.makedirs('./frame_image_save', exist_ok=True)  # (exist_ok=True) 폴더가 있으면 넘어감, 예외처리 대신 해준거
                cv2.imwrite('./frame_image_save/' + 'frame%d.jpg' % count, frame)
                print('save frame number >> ', str(int(cap.get(1))))
                count += 1
        else:
            break
else:
    print('can`t open video')

cap.release()
cv2.destroyAllWindows()