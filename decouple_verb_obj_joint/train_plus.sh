#!/bin/bash
python solve.py train_plus.prototxt ../verb_hand/deploy_plus.prototxt ../obj/obj_cropped/obj_deploy_plus44.prototxt PLUS44_data/RAHUL_train15.txt ../obj/obj_cropped/PLUS44_model/OBJ_PLUS44_RAHUL_iter_3000.caffemodel ../verb_hand/PLUS44_model15/PLUS44_RAHUL_T2_iter_8000.caffemodel PLUS44_model15/HAND_VERB_OBJ_PLUS44_RAHUL 4000 0 2>&1 | tee log/solve_plus44_rahul15.log
