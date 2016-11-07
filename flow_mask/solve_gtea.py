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
parser.add_argument('train_prototxt', help = 'Train prototxt file')
parser.add_argument('source', help = 'Source txt file')
parser.add_argument('model_prefix', help = 'Model file prefix')
parser.add_argument('base_lr', help = 'Base learning rate')
parser.add_argument('max_iterations', help = 'Max training iterations')
parser.add_argument('gpu_id', help = 'GPU ID')

#args = parser.parse_args()
args = argparse.Namespace()
args.gpu_id = 3
args.train_prototxt = 'train_gtea.prototxt'
args.source = 'GTEA_data/S4_train.txt'
args.model_prefix = 'GTEA_model/GTEA_S4'
args.hand_mean_file = 'GTEA_data/S4_mean.binaryproto'
verb_net_proto_file = 'verb_deploy_gtea.prototxt'

args.batch_size = 170
args.max_iterations = 5000
base_weights = "video__iter_10000.caffemodel"
solver_prototxt = 'solver.prototxt'

new_train_prototxt = set_properties(args.train_prototxt, 
    {
        'source' : args.source,
        'batch_size' : args.batch_size,
        'hand_mean_file' : args.hand_mean_file,
    })

new_solver_prototxt = set_properties(solver_prototxt, 
    {
        'net': new_train_prototxt,
        'snapshot_prefix': args.model_prefix,
    })

# init
caffe.set_mode_gpu()
caffe.set_device(int(args.gpu_id))

solver = caffe.SGDSolver(new_solver_prototxt)

verb_net = caffe.Net(verb_net_proto_file, base_weights, caffe.TEST)
for name in verb_net.params.keys():
    print 'copy {}...'.format(name)
    if name == 'conv1':
        solver.net.params[name][0].data[:,0:20,...] = verb_net.params[name][0].data
        solver.net.params[name][0].data[:,20:,...] = verb_net.params[name][0].data[:,0:10,...]
    else:
        solver.net.params[name][0].data[...] = verb_net.params[name][0].data

    solver.net.params[name][1].data[...] = verb_net.params[name][1].data

#solver.restore('PLUS44_model/PLUS44_RAHUL_iter_3000.solverstate')
solver.step(int(args.max_iterations))
