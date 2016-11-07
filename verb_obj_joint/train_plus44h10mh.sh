#!/bin/bash
python verb_obj_joint_solve.py verb_obj_joint_train_plus44.prototxt ../verb/verb_deploy_plus44.prototxt ../obj/obj_cropped/obj_deploy_plus44.prototxt PLUS44H10MH_data/AHMAD_train.txt ../obj/obj_cropped/PLUS44_model/OBJ_PLUS44_AHMAD_iter_3000.caffemodel ../verb/PLUS44H10MH_model/VERB_PLUS44H10MH_AHMAD_iter_3000.caffemodel PLUS44H10MH_model/VERB_OBJ_PLUS44H10MH_AHMAD 4000 1 2>&1 | tee log/verb_obj_joint_solve_plus44h10mh_ahmad.log