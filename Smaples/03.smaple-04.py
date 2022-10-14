# import qrcode
# from PIL import Image as img

# qr_data = 'www.naver.com'
# qr_image = qrcode.make(qr_data)

# qr_image.save(qr_data + '.png')

import qrcode
from PIL import Image as img

with open('site_list.txt','rt', encoding='UTF8') as f:
    read_lines = f.readlines()

for line in read_lines:
    line = line.strip() #글자외의 것들을 걸러줌
    # print(line)
    qr_data = line
    qr_image = qrcode.make(qr_data)
    qr_image.save(qr_data + '.png')