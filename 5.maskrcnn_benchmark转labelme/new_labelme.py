from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os, sys
import cv2
import datetime
from get_outline_json import get_outline_json
import json


config_file = '../../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
opts = ["MODEL.WEIGHT", "model_0182500.pth"]
# load config from file and command-line arguments
cfg.merge_from_file(config_file)
cfg.merge_from_list(opts)
cfg.freeze()

model = COCODemo(cfg,
                 min_image_size=800,
                 # confidence_threshold=0.7,
                 show_mask_heatmaps=False)  # 0-30

def get_mask_box(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray!=0] = 255
    contours,hierarchy = cv2.findContours(gray, 1, 2)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    return x,y,w,h

def read_mask(img_path):
    img = cv2.imread(img_path, 0)
    img[img!=0] = 1
    img = cv2.merge([img, img, img])
    return img


root_path = './images0408'
save_path = './0182500_save'


images_path = os.path.join(root_path)
images = os.listdir(images_path)

for image_name in images:
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)

    print(image_name)

    #x,y,w,h = get_mask_box('../masks/' + image_name[11:13] + '/label.png')
    #masks = read_mask('../masks/' + image_name[11:13] + '/label.png')

    #print(masks.shape)
    #image[masks==0] = 0
    #image = image[y:y+h, x:x+w, :]
    # 保存裁剪的数据

    cv2.imwrite(os.path.join(save_path, image_name), image)

    ret_img, json_data = get_outline_json(model, image, image_name)

    json_save_path = os.path.join(save_path)
    if not os.path.isdir(json_save_path):
        os.makedirs(json_save_path)


    #cv2.imwrite(os.path.join(save_path, image_name), image)
    with open(os.path.join(json_save_path, image_name[:-4]+'.json'),"w") as f:
        json.dump(json_data, f)
        print("write over")
    # break
