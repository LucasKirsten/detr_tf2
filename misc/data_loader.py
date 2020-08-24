import os
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.utils import to_categorical

def imread(path, resize=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img

def norm(image):
    return image/127.5 - 1.

def build_mask(image):
    return np.zeros(image.shape[:2], dtype=np.bool)
    
def generator(batch_size, path_to_data, num_classes, max_detections, num_decoder_layers=1, resize=None):

    path_images = os.path.join(path_to_data, 'images', '*.jpg')
    path_images = glob(path_images)

    # array for no object
    no_object = [0.]*(num_classes+1+4)
    no_object[0] = 1.

    while True:
        batchImg, batchMsk, batchY = [], [], []
        np.random.shuffle(path_images)
        
        for path_img in path_images:
            # verify if there is the path to label
            path_lab = path_img.replace('images', 'labels').replace('.jpg', '.txt')
            if not os.path.exists(path_lab):
                continue
            
            # open and normalize image
            image = norm(imread(path_img, resize))
            
            # create the output for labels
            labels = []
            label = open(path_lab, 'r').read().split('\n')
            for i in range(max_detections):
                if i>=len(label):
                    labels.append(no_object)
                    continue

                lab = label[i]
                if len(lab)<1: # verify for empty row
                    labels.append(no_object)
                    continue

                clas, cx,cy,w,h = lab.split(' ')

                clas = to_categorical(int(clas)+1, num_classes+1)
                labels.append([*clas, *map(float, (cx,cy,w,h))])
            
            batchImg.append(image); batchMsk.append(build_mask(image))
            batchY.append([labels]*num_decoder_layers)
            
            if len(batchImg)>=batch_size:
                yield (np.float32(batchImg), np.array(batchMsk)),\
                      (np.float32(batchY)[...,:-4], np.float32(batchY)[...,-4:])
                batchImg, batchMsk, batchY = [], [], []