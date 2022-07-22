# Classe usada para obter os dados de treino do modelo

from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
size = 300
contador = 0
pasta = "./data/V"

if not cap.isOpened():
    print("Erro ao acessar captura de vídeo")
else:   
    while cap.isOpened():
        ret, frame = cap.read()
        hands, frame = detector.findHands(frame)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            # print(hand['bbox'])
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

            else:
                k = size / w
                hCal = math.ceil(k * h)
                resize = cv2.resize(crop, (size, hCal))
                resize_shape = resize.shape
                hGap = math.ceil((size - hCal) / 2)
                mask[hGap:hCal + hGap, :] = resize

            # cv2.imshow('Crop', crop)
            # cv2.imshow('Output', mask)
        
        cv2.imshow('Frame', frame)

        if ret is True:
            key = cv2.waitKey(1)
            if(key == 27):
                break
            if (key == 32):
                contador += 1
                cv2.imwrite(f'{pasta}/Image_{time.time()}.jpg', mask)
                print(contador)

        else: break
