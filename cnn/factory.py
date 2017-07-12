import os
import cv2
from xml.etree.ElementTree import parse
import numpy as np
from sklearn.utils import shuffle
image_dir = '../dataset/object recognition/Pascal VOC/image'
anotation_dir = '../dataset/object recognition/Pascal VOC/anotation'

def make_train_dataset(num):
    #image
    image_list = os.listdir(image_dir)
    image_dic = {}
    for path in image_list:
        img = cv2.imread(os.path.join(image_dir, path))
        img = cv2.resize(img, (448, 448))

        image_dic[path.split('.')[0]] = img

    anotation_list = os.listdir(anotation_dir)
    anotation_list = shuffle(anotation_list)
    image = []
    anotation_cls1 = []
    anotation_cls2 = []
    anotation_reg1 = []
    anotation_reg2 = []
    count = 0
    for path in anotation_list:
        if count == num:
            break
        bbox1 = np.zeros((7, 7))
        bbox2 = np.zeros((7, 7, 4))
        bbox3 = np.zeros((7, 7))
        bbox4 = np.zeros((7, 7, 4))
        tree = parse(os.path.join(anotation_dir, path))
        root = tree.getroot()
        boxes = root.findall('object')
        if len(boxes) == 0:
            continue
        for box in boxes:
            box = box.find('bndbox')
            xmin = float(box.findtext('xmin'))
            xmax = float(box.findtext('xmax'))
            ymin = float(box.findtext('ymin'))
            ymax = float(box.findtext('ymax'))

            cx = (xmax - xmin) / 2
            cy = (ymax - ymin) / 2
            w = xmax - xmin
            h = ymax - ymin

            if bbox1[int(cx/64), int(cy/64)] == 0:
                bbox1[int(cx / 64), int(cy / 64)] = 1
                bbox2[int(cx / 64), int(cy / 64), 0] = cx
                bbox2[int(cx / 64), int(cy / 64), 1] = cy
                bbox2[int(cx / 64), int(cy / 64), 2] = w
                bbox2[int(cx / 64), int(cy / 64), 3] = h
            else:
                bbox3[int(cx / 64), int(cy / 64)] = 1
                bbox4[int(cx / 64), int(cy / 64), 0] = cx
                bbox4[int(cx / 64), int(cy / 64), 1] = cy
                bbox4[int(cx / 64), int(cy / 64), 2] = w
                bbox4[int(cx / 64), int(cy / 64), 3] = h


        image.append(image_dic[path.split('.')[0]])
        anotation_cls1.append(bbox1)
        anotation_cls2.append(bbox3)
        anotation_reg1.append(bbox2)
        anotation_reg2.append(bbox4)
        count += 1

    return np.array(image), np.array(anotation_cls1), np.array(anotation_reg1), np.array(anotation_cls2), np.array(anotation_reg2)
