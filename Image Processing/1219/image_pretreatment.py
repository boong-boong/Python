import os
import glob
import argparse
from PIL import Image


# image_pretreatment.py
# 오렌지 : Orange
# 자몽 : grapefruit
# 레드향 : Kanpei
# 한라봉 : Dekopon

# 폴더 구성 /dataset/image/각폴더명 생성/ 이미지 저장(resize 400 x 400)
# 직사각형 -> 정사각형 리사이즈 비율 유지
def expand2square(img, background_color):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), background_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    elif width < height:
        result = Image.new(img.mode, (height, height), background_color)
        result.paste(img, (0, (height - width) // 2))
        return result


def save_img(img_data):
    for i in img_data:
        print(i)
        file_name = i.split('/')
        dir_name = file_name[2]
        file_name = file_name[3]
        file_name = file_name.replace('.jpg', '.png')
        temp_img = Image.open(i)
        width, height = temp_img.size
        orange_img_resize = expand2square(temp_img, (0, 0, 0)).resize((400, 400))

        # 폴더 생성
        os.makedirs(f'./dataset/images/{dir_name}', exist_ok=True)
        orange_img_resize.save(f'./dataset/images/{dir_name}/{dir_name}_{file_name}')


def img_processing(orange_data, grapefruit_data, kanpei_data, dekopon_data):
    orange = orange_data
    grapefruit = grapefruit_data
    kanpei = kanpei_data
    dekopon = dekopon_data

    save_img(orange)
    save_img(grapefruit)
    save_img(kanpei)
    save_img(dekopon)


def image_file_check(opt):
    # image-folder-path
    # image_folder_path
    image_path = opt.image_folder_path
    # 각 폴더별 데이터 양 체크 #
    """
    image 
        - 자몽
            - xxx.jpg
        - 레드향
        - 한라봉 
    image_path -> ./image/orange/*.jpg
    """
    all_data = glob.glob(os.path.join(image_path, "*", "*.jpg"))
    print("전체 데이터 갯수 : ", len(all_data))
    # 오렌지
    ornage_data = glob.glob(os.path.join(image_path, "orange", "*.jpg"))
    print("오렌지 데이터 갯수 >> ", len(ornage_data))
    # 자몽
    grapefruit_data = glob.glob(os.path.join(
        image_path, "grapefruit", "*.jpg"))
    print("자몽 데이터 갯수 >> ", len(grapefruit_data))
    # 레드향
    kanpei_data = glob.glob(os.path.join(image_path, "kanpei", "*.jpg"))
    print("레드향 데이터 갯수 >> ", len(kanpei_data))
    # 한라봉
    dekopon_data = glob.glob(os.path.join(image_path, "dekopon", "*.jpg"))
    print("한라봉 데이터 갯수 >> ", len(dekopon_data))

    return ornage_data, grapefruit_data, kanpei_data, dekopon_data


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder-path", type=str, default="./images")
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    orange_data, grapefruit_data, kanpei_data, dekopon_data = image_file_check(opt)
    img_processing(orange_data, grapefruit_data, kanpei_data, dekopon_data)
