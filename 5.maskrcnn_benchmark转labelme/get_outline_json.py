from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os, sys
import cv2
import numpy as np
import pickle
import torch
import gc
import glob
import matplotlib.pyplot as plt
import random
import json


def find_dominant_points(contour,epsilon):
    '''
    Parameters:
        contour: one countourreturned from cv2.findContours
        epsilon: an accuracy parameter, maximum distance from contour to approximated contour
    Returns:
        approx: dominant points of contour, numpy array,each row is a x-y coordinate, exists in countour
    '''
    eps = epsilon*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,eps,True)
    approx = np.reshape(approx, (approx.shape[0], approx.shape[2]))
    return approx.tolist()


def get_outline_json(model, image, image_name):

    json_data = {
        "version": "3.5.0",
        "flags": {},
        "shapes": [
        ],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 255, 255, 255],
        "imagePath": "img.png",
        "imageData": None
    }

    cnt_json = {
        "label": "pig",
        "line_color": None,
        "fill_color": None,
        "points": [
        ],
        "shape_type": "polygon"
    }


    outlines = model.get_masks_for_cvimg(image)
    outlines = np.transpose(outlines, (0, 2, 3, 1))

    #print(outlines.shape)

    for idx, outline in enumerate(outlines):
        thresh = np.array(outline, dtype='uint8')

        #cv2.imwrite("./%d.png"%idx, outline)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        #print('contours', len(contours))
        cnt = [sorted(contours, key=cv2.contourArea, reverse=True)[0]]

        color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        cv2.drawContours(image, cnt, -1, color, 2)

        cnt = cnt[0].reshape((cnt[0].shape[0], cnt[0].shape[2]))
        #cnt = cnt.tolist()

        '''
        points = []
        for i in range(len(cnt)):
            skip_num = len(cnt)//20
            if i%skip_num==0:
                points.append(cnt[i])
        '''
        points = find_dominant_points(cnt, 7e-3)   #### test valeur de epsilon
        #print(len(points))



        cnt_json_cp = cnt_json.copy()
        cnt_json_cp["points"] = points
        cnt_json_cp["label"] = "pig%d"%idx
        json_data["shapes"].append(cnt_json_cp)

    json_data["imagePath"] = image_name
    cv2.putText(image, ('count: %d'%len(outlines)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    return image, json_data

