caffe_root = '/home/minghuam/caffe-fcn'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse
import pickle

def ls_images(folder_path, extension = '.png'):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('test_file', help = 'Test file')
parser.add_argument('deploy_file', help = 'Deploy file')
parser.add_argument('model_file', help = 'Model file')
parser.add_argument('output_dir', help = 'Output directory')
parser.add_argument('gpu_id', help = 'GPU id')
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(int(args.gpu_id))
net_proto_file =  args.deploy_file

net = caffe.Net(net_proto_file, args.model_file, caffe.TEST)

crop_size = 224
#Imean = np.load('image_mean.npy').transpose(2, 1, 0)
Imean = np.load('image_mean.npy')
x = (Imean.shape[1] - crop_size)/2
y = (Imean.shape[0] - crop_size)/2
Imean = Imean[y:y+crop_size, x:x+crop_size, :]

video_images = dict()
with open(args.test_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        img = tokens[0]
        label = int(tokens[1])
        video = os.path.basename(os.path.dirname(img))
        if video not in video_images:
            video_images[video] = dict()
            video_images[video]['images'] = list()
            video_images[video]['label'] = label
        video_images[video]['images'].append(img)

results = dict()
features = dict()
results['dataset'] = args.test_file.split('/')[0].split('_')[0]
results['subject'] = args.model_file.split('/')[-1].split('_')[2]
results['n_iterations'] = args.model_file.split('.')[-2].split('_')[-1]
for i, video in enumerate(video_images):
    print video, '{}/{}'.format(i+1, len(video_images))
    feat = dict()
    for img in video_images[video]['images']:
        I = cv2.imread(img)
        x = (I.shape[1] - crop_size)/2
        y = (I.shape[0] - crop_size)/2
        I = I[y:y+crop_size, x:x+crop_size, :]
        Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
        net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
        out = net.forward()
        feat[os.path.basename(img)] = net.blobs['fc8_gtea'].data.ravel()
    features[video] = (feat, label)

results['features'] = features
save_file = os.path.splitext(os.path.basename(args.test_file))[0] + '_{}_fc8.pkl'.format(results['n_iterations'])
pickle.dump(results, open(os.path.join(args.output_dir, save_file), 'w'))
