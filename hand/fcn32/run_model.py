caffe_root = '/home/minghuam/caffe-fcn'
import sys,os,shutil
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

def predict_one_image(net, Imean, imgs, K = 16):
    Iraw = np.zeros((K, 256, 256, 3), np.uint8)
    for i,img in enumerate(imgs):
        Itemp = cv2.imread(img)
        height, width = Itemp.shape[0:2]
        Iraw[i,...] = cv2.resize(Itemp, (256, 256))
    #Inorm = (I - Imean).reshape((K, I.shape[0], I.shape[1], I.shape[2]))
    Inorm = Iraw - Imean
    net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
    out = net.forward()

    e_score = np.exp(out['score'])
    e_score_sum = e_score.sum(axis = 1).reshape((1, K, 256, 256))
    e_score_sum = e_score_sum.transpose((1, 0, 2, 3)).repeat(repeats = 2, axis = 1)
    Iout =  e_score/e_score_sum
    Iout = (Iout[:,1,...]*255).astype(np.uint8).reshape((K, 256, 256, 1))
    Iout = np.repeat(Iout, repeats = 3, axis = 3)
    
    #Iout = (I1*255).astype(np.uint8).reshape(I1.shape[0], I1.shape[1], 1)
    #Iout = np.repeat(Iout, 3, axis = 2)
    #Iret = cv2.resize(Iout, (Iraw.shape[1], Iraw.shape[0]))

    return Iout, Iraw, width, height

parser = argparse.ArgumentParser()
parser.add_argument('images_dir', help = 'Images directory')
parser.add_argument('model_file', help = 'Model file')
parser.add_argument('output_dir', help = 'Output directory')
parser.add_argument('gpu_id', help = 'GPU id')
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(int(args.gpu_id))

net_proto_file =  'fcn32_deploy.prototxt'
net = caffe.Net(net_proto_file, args.model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

K = 32
Imean = np.ones((K, 256, 256, 3), np.float32)
for c in range(K):
    for i in range(3):
        Imean[c, :,:,i] *= bgr_mean[i]

if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

folders = sorted(os.listdir(args.images_dir))
for i,d in enumerate(folders):
    print '{}: {}/{}'.format(d, i+1, len(folders))
    os.mkdir(os.path.join(args.output_dir, d))
    images = sorted([os.path.join(args.images_dir, d, f) for f in os.listdir(os.path.join(args.images_dir, d))])
    start = 0
    n_images = len(images)
    while start < n_images:
        end = start + K
        if end > n_images:
            end = n_images

        Iout, Iraw, width, height = predict_one_image(net, Imean, images[start:end], K) 
        for index in range(start, end):
            i = index - start
            Iw = cv2.resize(Iout[i,...], (width, height))
            cv2.imwrite(os.path.join(args.output_dir, d, os.path.basename(images[index])), Iw)
            #Iw[:,:,0] = 0
            #Iw[:,:,2] = 0
            #Iret = cv2.addWeighted(cv2.resize(Iraw[i,...], (width, height)), 0.75, Iw, 0.5, 0, 0) 
            #cv2.imshow('Result', Iret)
            #if cv2.waitKey(10) & 0xFF == 27:
            #    break

        start = end
