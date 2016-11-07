#!/bin/bash
python fcn32_solve.py train_images.txt PROJECT_TRAIN_VOC 8000 0
#python fcn32_solve.py GTEA_data/s2_train.txt GTEA_model/FCN32_GTEA_S2 8000 0
python fcn32_solve.py GTEA_data/s3_train.txt GTEA_model/FCN32_GTEA_S3 8000 0
python fcn32_solve.py GTEA_data/s4_train.txt GTEA_model/FCN32_GTEA_S4 8000 0
