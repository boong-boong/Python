import cv2
import os
from xml.etree.ElementTree import parse
import albumentations as A

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # white


def find_xml_file(xml_folder_path):
    all_root = []
    for (path, dir, files) in os.walk(xml_folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".xml":
                root = os.path.join(path, filename)
                all_root.append(root)
    if len(all_root) == 0:
        print('no xml file')
    return all_root


def visualize_bbox(image, bboxes, category_ids, category_id_to_name,
                   color=BOX_COLOR, thickness=2):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        print("class_name >> ", class_name)
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_min + w), int(y_min), int(y_min + h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=color, thickness=thickness)
        cv2.putText(img, text=class_name, org=(x_min, y_min - 15),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color,
                    thickness=thickness)
    cv2.imshow("test", img)
    cv2.waitKey(0)


xml_dirs = find_xml_file('./annotations')

for xml_dir in xml_dirs:
    tree = parse(xml_dir)
    root = tree.getroot()
    img_metas = root.findall('image')

    for img_meta in img_metas:
        print(img_meta.attrib)
        img_name = img_meta.attrib['name']

        image_path = os.path.join('./', img_name)
        image = cv2.imread(image_path)
        # for y in img_meta:
        # print(y.tag, y.attrib)
        box_metas = img_meta.findall('box')
        for box_meta in box_metas:
            # print(box_meta.attrib)
            box_label = box_meta.attrib['label']
            box = [
                int(float(box_meta.attrib['xtl'])),
                int(float(box_meta.attrib['ytl'])),
                int(float(box_meta.attrib['xbr'])),
                int(float(box_meta.attrib['ybr'])),
            ]
            rect_img = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            rect_img = cv2.putText(rect_img, box_label, (box[0], box[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                   fontScale=1, color=(255, 255, 0),
                                   thickness=2)

        cv2.imshow('hw', image)
        cv2.waitKey(0)

transfrom = A.Compose([
    A.Resize(224, 224),
    A.RandomRotate90(p=0.7),
    A.HorizontalFlip(p=1)
])

transfromed = transfrom(image=image)
cv2.imshow('test', transfromed['image'])
cv2.waitKey(0)
