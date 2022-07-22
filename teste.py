from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier('./model/keras_model.h5', './model/labels.txt')

offset = 20
size = 300
contador = 0

# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# labels = ['A', 'C', 'E', 'F', 'M', 'N', 'R', 'U', 'V']
labels = ['A', 'B', 'C', 'E', 'F', 'M', 'N', 'R', 'U', 'V']

if not cap.isOpened():
    print("Erro ao acessar captura de vídeo")
else:   
    while cap.isOpened():
        ret, frame = cap.read()
        output = frame.copy()
        hands, frame = detector.findHands(frame)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            print(hand['bbox'])
            # mask = np.zeros((size, size, 3), np.uint8)
            mask = np.ones((size, size, 3), np.uint8) * 255
            crop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            crop_shape = crop.shape

            aspect = h / w

            if aspect > 1:
                k = size / h
                wCal = math.ceil(k * w)
                resize = cv2.resize(crop, (wCal, size))
                resize_shape = resize.shape
                wGap = math.ceil((size - wCal) / 2)
                mask[:, wGap:wCal + wGap] = resize
                prediction, index = classifier.getPrediction(mask, draw=False)
                print(prediction, index)

            else:
                k = size / w
                hCal = math.ceil(k * h)
                resize = cv2.resize(crop, (size, hCal))
                resize_shape = resize.shape
                hGap = math.ceil((size - hCal) / 2)
                mask[hGap:hCal + hGap, :] = resize
                prediction, index = classifier.getPrediction(mask, draw=False)

            cv2.putText(output, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            # cv2.rectangle(output, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # cv2.imshow('Crop', crop)
            # cv2.imshow('Output', mask)
        
        cv2.imshow('Frame', output)

        if ret is True:
            key = cv2.waitKey(1)
            if(key == 27):
                break
        else: break
