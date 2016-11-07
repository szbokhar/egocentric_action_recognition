#!/bin/bash
python verb_obj_joint_solve.py verb_obj_joint_train_gaze25.prototxt ../verb/verb_deploy_gaze25.prototxt ../obj/obj_cropped/obj_deploy_gaze25.prototxt GAZE25_data/G1_train.txt ../obj/obj_cropped/GAZE25_model/OBJ_GAZE25_iter_3000.caffemodel ../verb/GAZE25_model/VERB_GAZE25_iter_2000.caffemodel GAZE25_model/VERB_OBJ_GAZE25_G1 4000 3 2>&1 | tee log/verb_obj_joint_solve_gaze25.log
