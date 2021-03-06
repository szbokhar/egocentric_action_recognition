caffe_root = '/home/minghuam/caffe-fcn'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse

def ls_images(folder_path, extension = '.png'):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)

subject = 'S2'
#folders = [d for d in os.listdir(input_dir) if d.lower().startswith(subject)]
input_dir = '/home/minghuam/data/GTEA/RAW/raw_images_large'
output_dir = 'GTEA_output2/{}'.format(subject)
prob_output_dir = 'GTEA_output2/{}_prob'.format(subject)
net_proto_file =  'fcn32_obj_deploy.prototxt'
model_file = 'GTEA_model/FCN32_OBJ_GTEA_{}_iter_4000.caffemodel'.format(subject)

folders = [d for d in os.listdir(input_dir)]
print folders

mkdir_safe(output_dir)
#mkdir_safe(prob_output_dir)
caffe.set_mode_gpu()
caffe.set_device(1)

net = caffe.Net(net_proto_file, model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

crop_size = 224
Imean = np.ones((crop_size, crop_size, 3), np.float32)
for i in range(3):
    Imean[:,:,i] *= bgr_mean[i]

#cap = cv2.VideoCapture(0)
#fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
#video_writer = cv2.VideoWriter('./hand_detection.avi', fourcc, 25, (360, 203), True)
# if not video_writer.isOpened():
#   print 'not opened'
#   sys.exit(0)

for folder in folders:
    mkdir_safe(os.path.join(output_dir, folder))
    #mkdir_safe(os.path.join(prob_output_dir, folder))
    images = ls_images(os.path.join(input_dir, folder), '.jpg')
    last_left = -1
    last_top = -1
    for img in images:
        print img
        Iraw = cv2.imread(img)
        I = cv2.resize(Iraw, (256, 256))
        x = (I.shape[1] - crop_size)/2
        y = (I.shape[0] - crop_size)/2
        I = I[y:y+crop_size, x:x+crop_size,:]
        Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
        net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
        out = net.forward()

        e_score = np.exp(out['score'])[0,:,:,:]
        e_score_sum = e_score.sum(axis = 0)
        Iobj =  e_score[1,:,:]/e_score_sum
        Iobj = (Iobj*255).astype(np.uint8)
        Iobj = cv2.resize(Iobj, (360, 203))

        unused, Ithresh = cv2.threshold(Iobj, 64, 255, 0)
        #cv2.imshow('Ithresh', Ithresh)

        contours, hierarchy = cv2.findContours(Ithresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_index = -1
        max_area = -1
        min_area = 100
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if max_area < area:
                max_index = i
                max_area = area

        left = -1
        top = -1
        obj_size = 120
        if max_index != -1:
            M = cv2.moments(contours[max_index])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            top = cy - obj_size/2
            bottom = cy + obj_size/2 - 1
            if top < 0:
                top = 0
            if bottom > Iobj.shape[0] - 1:
                top = Iobj.shape[0] - 1 - obj_size

            left = cx - obj_size/2
            right = cx + obj_size/2 - 1
            if left < 0:
                left = 0
            if right > Iobj.shape[1] - 1:
                left= Iobj.shape[1] - 1 - obj_size

        # cv2.imshow('Ithresh', Ithresh)
        # cv2.imshow('Iprob', Iobj)
        
        Iobj = np.repeat(Iobj.reshape(Iobj.shape + (1,)), 3, axis = 2)
        Iobj[:,:,0] = 0
        Iobj[:,:,1] = 0

        if left == -1:
            if last_left != -1:
                left = last_left
                top = last_top
            else:
                left = Ithresh.shape[1]/2 - obj_size/2
                top = Ithresh.shape[0]/2 - obj_size/2
        last_left = left
        last_top = top

        cv2.rectangle(Iobj, (left, top), (left + obj_size, top + obj_size), (0, 255, 0), 1)
        y = int(Iraw.shape[0] * float(top)/Iobj.shape[0])
        x = int(Iraw.shape[1] * float(left)/Iobj.shape[1])
        size = int(Iraw.shape[0] * float(obj_size)/Iobj.shape[0])
        if x + size > Iraw.shape[1] - 1:
            x = Iraw.shape[1] - 1 - size
        if y + size > Iraw.shape[0] - 1:
            y = Iraw.shape[0] - 1 - size
        Icrop = Iraw[y:y+size, x:x+size, :]
        Icrop = cv2.resize(Icrop, (256, 256))
                    #cv2.imwrite(os.path.join(prob_output_dir, folder, os.path.basename(img)), Iobj)
        cv2.imwrite(os.path.join(output_dir, folder, os.path.basename(img)), Icrop)
        #cv2.imshow('Iobj', Icrop)

        I = cv2.resize(I, (360, 203))
        Iret = cv2.addWeighted(I, 0.75, Iobj, 0.5, 0, 0)
        #cv2.imwrite(os.path.join(prob_output_dir, folder, os.path.basename(img)), Iret)

        #video_writer.write(Iret)
        cv2.imshow("ret", Iret)

        if cv2.waitKey(10) & 0xFF == 27:
            sys.exit(0)

#video_writer.release()

