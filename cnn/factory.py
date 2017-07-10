import os
import cv2
from xml.etree.ElementTree import parse
import numpy as np
image_dir = '../dataset/object recognition/Pascal VOC/image'
anotation_dir = '../dataset/object recognition/Pascal VOC/anotation'

def make_train_dataset():
    #image
    image_list = os.listdir(image_dir)
    image_dic = {}
    for path in image_list:
        img = cv2.imread(os.path.join(image_dir, path))
        img = cv2.resize(img, (400, 400))

        image_dic[path.split('.')[0]] = img

    anotation_list = os.listdir(anotation_dir)
    image = []
    anotation = []
    for path in anotation_list:
        tree = parse(os.path.join(anotation_dir, path))
        root = tree.getroot()
        box = root.find('object').find('bndbox')
        if box == None:
            continue
        xmin = box.findtext('xmin')
        xmax = box.findtext('xmax')
        ymin = box.findtext('ymin')
        ymax = box.findtext('ymax')

        cx = (xmax - xmin) / 2
        cy = (ymax - ymin) / 2
        w = xmax - xmin
        h = ymax - ymin

        image.append(image_dic[path.split('.')[0]])
        anotation.append([cx, cy, w, h])

    return np.array(image), np.array(anotation)