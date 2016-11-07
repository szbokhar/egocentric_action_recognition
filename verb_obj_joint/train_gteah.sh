#!/bin/bash
python verb_obj_joint_solve.py verb_obj_joint_train_gteah.prototxt ../verb/verb_deploy_gteah.prototxt ../obj/obj_cropped/obj_deploy_gteah.prototxt GTEAH_data/S2_train.txt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S2_iter_3000.caffemodel ../verb/GTEAH_model/VERB_GTEAH_S2_iter_2000.caffemodel GTEAH_model/VERB_OBJ_GTEAH_S2 4000 0 2>&1 | tee log/verb_obj_joint_solve_gteah_s2.log
