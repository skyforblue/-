import cv2
import os
import numpy as np
import threading

video_folder1 = '/home/ahc/20190812/Bar036/'
video_folder2 = '/home/ahc/20190812/Bar037/'
video_folder3 = '/home/ahc/20190812/Bar038/'
video_folder4 = '/home/ahc/20190812/Bar039/'
video_folder5 = '/home/ahc/20190812/Bar040/'
save = '/home/ahc/res/'

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

def seam_video(video_path, save_path):
    print(video_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_name = save_path.split('/')[-2]

    times = 0
    img_num = 0
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    _, image = cap.read()
    if _:
        x = int(image.shape[0] / 2)
        y = int(image.shape[1] / 2)
        compare_ = image[x - 400:x + 400, y - 400:y + 400]

    compare_data = cv2.cvtColor(compare_, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        flag, image = cap.read()
        if not flag:
            break

        if times % FPS == 0:
            image = image[x - 400:x + 400, y - 400:y + 400]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = compare_image(compare_data, gray_image, 12)
            if score < 90:
                cv2.imwrite(save_path + img_name + '_' + str(img_num) + '.png', image)
                img_num += 1
        times += 1
    cap.release()


def seam(video_folder, save):
    file_names = os.listdir(video_folder)
    video_names = [file for file in file_names if file.endswith('.mp4')]
    video_names.sort()

    for video_name in video_names:
        video_path = os.path.join(video_folder, video_name)
        bar_num = video_folder.split('/')[-2]
        video_name = video_name[:-4]
        save_path = os.path.join(save, bar_num, video_name) + '/'
        seam_video(video_path, save_path)


threads = []
threads.append(threading.Thread(target=seam, args=(video_folder1, save,)))
threads.append(threading.Thread(target=seam, args=(video_folder2, save,)))
threads.append(threading.Thread(target=seam, args=(video_folder3, save,)))
threads.append(threading.Thread(target=seam, args=(video_folder4, save,)))
threads.append(threading.Thread(target=seam, args=(video_folder5, save,)))
for t in threads:
    t.start()
for t in threads:
    t.join()



