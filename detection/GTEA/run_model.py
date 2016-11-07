caffe_root = '/home/minghuam/caffe-dev/'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse
import random
from collections import deque
import pickle


if __name__ == '__main__':

    test_file = '/home/minghuam/egocentric_action_recognition/detection/GTEA/data/S1_Tea_C1.txt'
    deploy_prototxt = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_gtea.prototxt'
    model_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_model/VERB_GTEA_S1_iter_2000.caffemodel'
    gpu_id = 3

    caffe.set_mode_gpu()
    caffe.set_device(int(gpu_id))
    net_proto_file = deploy_prototxt
    net = caffe.Net(net_proto_file, model_file, caffe.TEST)

    frames = list()
    with open(test_file, 'r') as fr:
        for line in fr.readlines():
            tokens = line.strip().split(' ')
            frames.append((tokens[0], tokens[1], int(tokens[2])))

    crop_size = 224
    num_stack_frames = 10
    num_total_frames = len(frames)
    mean = 128.0
    Iq_x = deque()
    Iq_y = deque()
    data = np.zeros((1, num_stack_frames * 2, crop_size, crop_size), np.float64)
    
    print 'total_frames:', num_total_frames
    index = 0
    count = 0
    scores = dict()
    while True:
        while len(Iq_x) < num_stack_frames:
            Ix = cv2.imread(frames[index][0])
            Iy = cv2.imread(frames[index][1])
            x = (Ix.shape[1] - crop_size)/2
            y = (Ix.shape[0] - crop_size)/2
            Ix = Ix[y:y+crop_size, x:x+crop_size, :]
            Iy = Iy[y:y+crop_size, x:x+crop_size, :]
            Ix = Ix[...,0].astype(np.float64) - mean
            Iy = Iy[...,0].astype(np.float64) - mean
            Iq_x.append(Ix)
            Iq_y.append(Iy)
            index += 1

        for i, (Ix, Iy) in enumerate(zip(Iq_x, Iq_y)):
            data[0,i*2+0,...] = Ix
            data[0,i*2+1,...] = Iy
        net.blobs['data'].data[...] = data
        score = net.forward()['score'].ravel()
        basename = os.path.basename(frames[index-num_stack_frames][0])
        label = frames[index-num_stack_frames][2]
        scores[basename] = (score.copy(), label)

        Iq_x.popleft()
        Iq_y.popleft()

        if index > num_total_frames - 1:
            break

    save_file = os.path.basename(test_file).split('.')[0] + '_results.pkl'
    pickle.dump(scores, open(save_file, 'w'))