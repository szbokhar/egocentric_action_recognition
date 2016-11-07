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

def predict_one_video(net, bx_images, by_images, hx_images, hy_images, data_type):
    crop_size = 224
    num_stack_frames = 10
    mean = 128.0
    Iq = deque()

    prob = None
    num_runs = 0
    for (bx_img, by_img, hx_img, hy_img) in zip(bx_images, by_images, hx_images, hy_images):
        Is = list()
        if data_type == 0 or data_type == 2:
            Is.append(cv2.imread(bx_img))
            Is.append(cv2.imread(by_img))
        if data_type == 1 or data_type == 2:
            Is.append(cv2.imread(hx_img))
            Is.append(cv2.imread(hy_img))

        for i in range(len(Is)):
            I = Is[i]
            x = (I.shape[1] - crop_size)/2
            y = (I.shape[0] - crop_size)/2
            Is[i] = I[y:y+crop_size, x:x+crop_size, 0].astype(np.float64) - mean

        Iq.append(Is)
        
        if len(Iq) == num_stack_frames:
            chs = num_stack_frames * 2
            if data_type == 2:
                chs = num_stack_frames * 4
            data = np.zeros((1, chs, crop_size, crop_size), np.float64)
            for i, Is in enumerate(Iq):
                chs = len(Is)
                for j in range(chs):
                    data[0,i*chs+j,...] = Is[j]
            net.blobs['data'].data[...] = data
            
            score = net.forward()['score'].ravel()
            #net.forward()
            #score = net.blobs['fc7'].data.ravel()

            num_runs += 1
            
            if prob is None:
                prob = score
            else:
                prob = prob + score
            
            Iq.popleft()
            
    return prob/num_runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy_prototxt', help = 'deploy prototxt')
    parser.add_argument('data_type', help = 'data type')
    parser.add_argument('test_file', help = 'Test file')
    parser.add_argument('model_file', help = 'Model file')
    parser.add_argument('output_dir', help = 'Output directory')
    parser.add_argument('gpu_id', help = 'GPU id')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    net_proto_file = args.deploy_prototxt

    net = caffe.Net(net_proto_file, args.model_file, caffe.TEST)

    test_videos = dict()
    with open(args.test_file, 'r') as fr:
        for line in fr.readlines():
            tokens = line.strip().split(' ')
            test_videos[tokens[0]] = int(tokens[1])

    predict_results = dict()
    scores = dict()
    results = dict()
    results['dataset'] = args.test_file.split('/')[0].split('_')[0]
    results['subject'] = args.model_file.split('/')[-1].split('_')[2]
    results['n_iterations'] = args.model_file.split('.')[-2].split('_')[-1]
    print results
    for i, video in enumerate(test_videos):
        bx_images = ls_images(os.path.join(video, 'bx'), '.jpg')
        by_images = ls_images(os.path.join(video, 'by'), '.jpg')
        hx_images = ls_images(os.path.join(video, 'hx'), '.jpg')
        hy_images = ls_images(os.path.join(video, 'hy'), '.jpg')
        label = test_videos[video]
        score = predict_one_video(net, bx_images, by_images, hx_images, hy_images, int(args.data_type))
        scores[os.path.basename(video)] = (score, label)
        predict_label = np.argmax(score)
        #predict_label = predict_one_video2(net, flow_x_images, flow_y_images)
        
        print os.path.basename(video), label, predict_label
        predict_results[video] = (label, predict_label)

    save_file = os.path.splitext(os.path.basename(args.test_file))[0] + '_{}_results.pkl'.format(results['n_iterations'])
    results['scores'] = scores
    pickle.dump(results, open(os.path.join(args.output_dir, save_file), 'w'))

    # overall accuracy
    correct_count = 0
    for video in predict_results:
        if predict_results[video][0] == predict_results[video][1]:
            correct_count += 1
    print 'overall accuracy:',float(correct_count)/len(predict_results)

    per_action_results = dict()
    for video in predict_results:
        action = '_'.join(video.split('_')[1:-1])
        if action not in per_action_results:
            per_action_results[action] = list()
        per_action_results[action].append(predict_results[video])

    per_action_accuracy = dict()
    for action in per_action_results:
        total_count = len(per_action_results[action])
        correct_count = 0
        for (label, predict_label) in per_action_results[action]:
            if label == predict_label:
                correct_count += 1
        per_action_accuracy[action] = float(correct_count)/total_count

    accuracies = per_action_accuracy.values()
    print 'average accuracy:', sum(accuracies)/len(accuracies)
