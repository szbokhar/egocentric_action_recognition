#!/bin/bash
python verb_obj_joint_solve.py verb_obj_joint_train_gaze.prototxt ../verb/verb_deploy_gaze.prototxt ../obj/obj_cropped/obj_deploy_gaze.prototxt GAZE_data/G1_train.txt ../obj/obj_cropped/GAZE_model/OBJ_GAZE_iter_3000.caffemodel ../verb/GAZE_model/VERB_GAZE_iter_2000.caffemodel GAZE_model/VERB_OBJ_GAZE_G1 4000 2 2>&1 | tee verb_obj_joint_solve_gaze.log
