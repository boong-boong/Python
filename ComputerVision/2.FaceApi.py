import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscription_key = '37e6e71fc56a4a6880fc6a4387b0afc6'
face_api_url = 'https://labuser58face.cognitiveservices.azure.com/face/v1.0/detect'
# Class, Library, Package 대문자 관례
# 지역변수, 파라메타 소문자 관례
# addr, msg 줄임알 배제 줄임은 옛날 메모리 아끼려고 현재는 의미가 명확한게 좋음
# 두 단어가 합쳐지면 두 번째 단어는 대문자
# 상수는 전체가 대문자 const MAX_USER = 100


image_url = 'https://file2.nocutnews.co.kr/newsroom/image/2022/05/02/202205021943360306_0.jpg'

image = Image.open(BytesIO(requests.get(image_url).content))

headers = {'Ocp-Apim-Subscription-key' : subscription_key}

params = {
    'returnFaceID': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'Smile'
}

data = {'url': image_url}

response = requests.post(face_api_url, params=params, headers=headers,json=data)
faces = response.json()
#faces

draw = ImageDraw.Draw(image)

def DrawBox(faces):

  for face in faces:
    #print(face)
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    width = rect['width']
    height = rect['height']

    draw.rectangle(((left, top),(left + width, top + height)), outline='red')

    face_attributes = face['faceAttributes']
    smile = face_attributes['smile']

    draw.text((left, top), str(smile), fill = 'red') # draw는 글자크기 조정 옵션없음

DrawBox(faces)