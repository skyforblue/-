import os,glob
import cv2
import numpy as np
# 相似度检测
def compare_image(image1, image2, size):
    sign1 = average_Hash(image1, size)
    sign2 = average_Hash(image2, size)

    score = 0
    for i in range(0, len(sign1)):
        if sign1[i] == sign2[i]:
            score += 1
        else:
            pass

    return int(score / len(sign1) * 100)

def average_Hash(image, size):
    resize_img = cv2.resize(image, (size, size))
    average_gray = resize_img.sum() / (size * size)
    img_value = np.array(resize_img).flatten()

    sign = ''
    for value in img_value:
        if value > average_gray:
            sign += '1'
        else:
            sign += '0'

    return sign
compare_ = cv2.imread('/home/yang/Desktop/产房/36/pic_192.168.102.136_3649.jpg')
compare_data = cv2.cvtColor(compare_, cv2.COLOR_BGR2GRAY)
for image in glob.glob("/home/yang/Desktop/产房/36/*.jpg"):
    frame = cv2.imread(image)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = compare_image(compare_data, gray_image, 10)
    if score>91 and score<100:
        os.remove(image)
        print("del ",image)