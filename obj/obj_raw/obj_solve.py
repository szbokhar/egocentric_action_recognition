caffe_root = '/home/minghuam/caffe-dev/'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import argparse
import numpy as np

def set_properties(prototxt, properties):
    print properties
    basename = os.path.basename(prototxt)
    with open(prototxt, 'r') as fr:
        lines = fr.readlines()
    for i, line in enumerate(lines):
        for key in properties:
            if line.strip().startswith(key + ':'):
                index = line.index(':')
                if type(properties[key]) == str:
                    lines[i] = line[:index+1] + ''' "''' + properties[key] + '''"\n'''
                else:
                    lines[i] = line[:index+1] + ' ' + str(properties[key]) + '\n'
    new_prototxt = '.' + prototxt
    with open(new_prototxt, 'w') as fw:
        for line in lines:
            fw.write(line)
    return new_prototxt


parser = argparse.ArgumentParser()
parser.add_argument('train_prototxt', help = 'Train prototxt')
parser.add_argument('source', help = 'Source txt file')
parser.add_argument('model_prefix', help = 'Model file prefix')
parser.add_argument('max_iterations', help = 'Max training iterations')
parser.add_argument('gpu_id', help = 'GPU id')
args = parser.parse_args()

base_weights = "VGG_CNN_M.caffemodel"
train_prototxt = args.train_prototxt
solver_prototxt = 'obj_solver.prototxt'
new_train_prototxt = set_properties(train_prototxt, {'source' : args.source})
new_solver_prototxt = set_properties(solver_prototxt, 
    {
        'net': new_train_prototxt,
        'snapshot_prefix': args.model_prefix
    })

# init
caffe.set_mode_gpu()
caffe.set_device(int(args.gpu_id))

solver = caffe.SGDSolver(new_solver_prototxt)
solver.net.copy_from(base_weights)

#solver.step(2000)
solver.step(int(args.max_iterations))
