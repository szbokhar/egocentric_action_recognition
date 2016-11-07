import os,sys,shutil

flow_dir = '/home/minghuam/data/GTEA/RAW/raw_flow'
label_flow_dir = '/home/minghuam/data/GTEA/RAW/flow/'
verb_id_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_data/verb_ids.txt'
output_dir = 'data'

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)

verb_ids = dict()
with open(verb_id_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		verb_ids[tokens[0]] = int(tokens[1])

image_verb_ids = dict()
for d in os.listdir(label_flow_dir):
	verb = d.split('_')[1]
	verb_id = verb_ids[verb]
	for img in os.listdir(os.path.join(label_flow_dir, d, 'x')):
		image_verb_ids[img] = verb_id

for d in os.listdir(flow_dir):
	print d
	with open(os.path.join(output_dir, d + '.txt'), 'w') as fw:
		for img in sorted(os.listdir(os.path.join(flow_dir, d, 'x'))):
			verb_id = -1
			if img in image_verb_ids:
				verb_id = image_verb_ids[img]
			fw.write(os.path.join(flow_dir, d, 'x', img) + ' ' + os.path.join(flow_dir, d, 'y', img) + ' ' + str(verb_id) + '\n')