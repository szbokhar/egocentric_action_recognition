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

def ls_images(folder_path, extension = '.png'):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])


def sample_motion_features(net, layer_name, flow_x_images, flow_y_images):
    crop_size = 224
    num_stack_frames = 10
    mean = 128.0

    Iq_x = deque()
    Iq_y = deque()

    features = dict()
    for index, (flow_x_img, flow_y_img) in enumerate(zip(flow_x_images, flow_y_images)):
        #print flow_x_img, flow_y_img
        Ix = cv2.imread(flow_x_img)
        Iy = cv2.imread(flow_y_img)
        
        x = (Ix.shape[1] - crop_size)/2
        y = (Ix.shape[0] - crop_size)/2

        Ix = Ix[y:y+crop_size, x:x+crop_size, :]
        Iy = Iy[y:y+crop_size, x:x+crop_size, :]

        Ix = Ix[...,0].astype(np.float64) - mean
        Iy = Iy[...,0].astype(np.float64) - mean

        Iq_x.append(Ix)
        Iq_y.append(Iy)

        if len(Iq_x) == num_stack_frames:
            data = np.zeros((1, num_stack_frames * 2, crop_size, crop_size), np.float64)
            for i, (Ix, Iy) in enumerate(zip(Iq_x, Iq_y)):
                data[0,i*2+0,...] = Ix
                data[0,i*2+1,...] = Iy
            net.blobs['data'].data[...] = data
            score = net.blobs[layer_name].data.ravel()
            basename = os.path.basename(flow_x_images[index - num_stack_frames + 1])
            features[basename] = score

            Iq_x.popleft()
            Iq_y.popleft()
    

    return features

def sample_object_features(net, layer_name, Imean, images):
    crop_size = 224
    x = (Imean.shape[1] - crop_size)/2
    y = (Imean.shape[0] - crop_size)/2
    Imean = Imean[y:y+crop_size, x:x+crop_size, :]
    features = dict()
    for img in images:
        I = cv2.imread(img)
        x = (I.shape[1] - crop_size)/2
        y = (I.shape[1] - crop_size)/2
        I = I[y:y+crop_size, x:x+crop_size, :]
        Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
        net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
        out = net.forward()
        features[os.path.basename(img)] = net.blobs[layer_name].data.ravel()
    return features

def mkdir_safe(d):
    if os.path.exists(d):
        return
    os.mkdir(d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--object', help = 'Object feature', action = 'store_true')
    parser.add_argument('test_file', help = 'Test file')
    parser.add_argument('deploy_file', help = 'Deploy file')
    parser.add_argument('model_file', help = 'Model file')
    parser.add_argument('output_dir', help = 'Output directory')
    parser.add_argument('gpu_id', help = 'GPU id')
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    net = caffe.Net(args.deploy_file, args.model_file, caffe.TEST)
    Imean = np.load('image_mean.npy')

    mkdir_safe(args.output_dir)
    feature_folder = os.path.join(args.output_dir, os.path.basename(args.test_file).split('.')[0])
    mkdir_safe(feature_folder)
    if args.object:
        object_feature_folder = os.path.join(feature_folder, 'object_fc8')
        mkdir_safe(object_feature_folder)
    else:
        motion_feature_folder = os.path.join(feature_folder, 'motion_fc8')
        mkdir_safe(motion_feature_folder)

    flow_folders = []
    object_folders = []
    with open(args.test_file, 'r') as fr:
        for line in fr.readlines():
            tokens = line.split(' ')
            flow_folders.append(tokens[0])
            object_folders.append(tokens[1])

    print '{} folders'.format(len(flow_folders))

    if args.object:
        for i, folder in enumerate(object_folders):
            print '{}: {}/{}'.format(folder, i+1, len(object_folders))
            images = sorted([os.path.join(folder, img) for img in os.listdir(folder)])
            features = sample_object_features(net, 'fc8_gtea', Imean, images)
            save_file = os.path.join(object_feature_folder, os.path.basename(folder) + '.pkl')
            pickle.dump(features, open(save_file, 'w'))

    else:
        for i, folder in enumerate(flow_folders):
            print '{}: {}/{}'.format(folder, i+1, len(flow_folders))
            x_folder = os.path.join(folder, 'x')
            y_folder = os.path.join(folder, 'y')
            x_images = sorted([os.path.join(x_folder, img) for img in os.listdir(x_folder)])
            y_images = sorted([os.path.join(y_folder, img) for img in os.listdir(y_folder)])
            features = sample_motion_features(net, 'fc8_gtea', x_images, y_images)
            save_file = os.path.join(motion_feature_folder, os.path.basename(folder) + '.pkl')
            pickle.dump(features, open(save_file, 'w'))
