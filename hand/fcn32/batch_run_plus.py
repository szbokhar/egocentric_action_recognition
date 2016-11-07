import os,sys
from multiprocessing import Process
import subprocess

images_dir = '/home/minghuam/data/GTEA_gaze_plus/RAW/raw_images_resized'
model_file = 'PLUS44_model/FCN32_PLUS44_AHMAD_iter_8000.caffemodel'
output_dir = 'PLUS44_output/AHMAD_output/'
gpu_ids = [1,2,3]

def proc(model_file, gpu_id, images_dir, output_dir):
	subprocess.call(['python', 'run_model_plus.py', images_dir, model_file, output_dir, str(gpu_id)])

folders = sorted([os.path.join(images_dir, d) for d in os.listdir(images_dir)])
num_folders = len(folders)
num_procs = len(gpu_ids)
index = 0
while index < num_folders:
    procs = []
    for i in range(index, index + num_procs):
        if i < num_folders:
            folder = folders[i]
            print folder, '{}/{}'.format(i+1, num_folders)
            p = Process(target = proc, args = (model_file, gpu_ids[i%3], folder, os.path.join(output_dir, os.path.basename(folder))))
            procs.append(p)
            p.start()
    for p in procs:
        p.join()
    index += num_procs