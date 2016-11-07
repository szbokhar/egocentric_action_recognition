import os
import shutil
import random
import cv2

def ls_images(folder_path, extension = '.png'):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mkdir_new(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

HAS_MANIPULATION_DATA = True

gt_folder = '/home/minghuam/data/GTEA/RAW/HandMask/GroundTruth'
img_folder = '/home/minghuam/data/GTEA/RAW/HandMask/Images'
mp_file = '/home/minghuam/data/GTEA/mp.txt'
data_folder = 'GTEA_data'
data_gt_folder = os.path.join(data_folder, 'gt')
data_img_folder = os.path.join(data_folder, 'images')

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

if not os.path.exists(data_gt_folder):
    os.mkdir(data_gt_folder)

if not os.path.exists(data_img_folder):
    os.mkdir(data_img_folder)

images = ls_images(img_folder, '.jpg')
gt_images = ls_images(gt_folder, '.png')

mp_points = dict()
with open(mp_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        basename = tokens[0]
        mp_x = int(tokens[1])
        mp_y = int(tokens[2])
        mp_points[basename] = (mp_x, mp_y)

subject_image_pairs = dict()
for (img, gt_img) in zip(images, gt_images):
    basename = os.path.basename(img)
    tokens = basename.split('_')
    subject = tokens[0].lower()
    if subject not in subject_image_pairs:
        subject_image_pairs[subject] = list()
    subject_image_pairs[subject].append((img, gt_img) + mp_points[basename])

print subject_image_pairs.keys()

input_width = 256
input_height = 256

for subject in subject_image_pairs:
    with open(os.path.join(data_folder, subject + '_test.txt'), 'w') as fw:
        for pair in subject_image_pairs[subject]:
            basename = os.path.basename(pair[0])
            I = cv2.imread(pair[0])
            height,width = I.shape[:2]
            I = cv2.resize(I, (input_width, input_height))
            Igt = cv2.resize(cv2.imread(pair[1]), (input_width, input_height))
            img = os.path.join(data_img_folder, os.path.basename(pair[0]))
            gt = os.path.join(data_gt_folder, os.path.basename(pair[1]))
            cv2.imwrite(img, I)
            cv2.imwrite(gt, Igt)
            line = img + " " + gt + " " + \
               str(mp_points[basename][0]/float(width)) + " " + \
               str(mp_points[basename][1]/float(height)) + "\n"
            fw.write(line)

    with open(os.path.join(data_folder, subject + '_train.txt'), 'w') as fw:
        for other_subject in subject_image_pairs:
            if subject != other_subject:
                for pair in subject_image_pairs[other_subject]:
                    basename = os.path.basename(pair[0])
                    I = cv2.imread(pair[0])
                    height,width = I.shape[:2]
                    Igt = cv2.resize(cv2.imread(pair[1]), (input_width, input_height))
                    if Igt.sum() == 0:
                        continue            
                    img = os.path.join(data_img_folder, os.path.basename(pair[0]))
                    gt = os.path.join(data_gt_folder, os.path.basename(pair[1]))
                    line = img + " " + gt + " " + \
                       str(mp_points[basename][0]/float(width)) + " " + \
                       str(mp_points[basename][1]/float(height)) + "\n"
                    #line = img + ' ' + gt + ' ' + hand_img + '\n'
                    fw.write(line)
