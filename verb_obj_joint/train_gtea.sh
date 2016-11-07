#!/bin/bash
#python verb_obj_joint_solve.py GTEA_data/S1_train.txt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S1_iter_3000.caffemodel ../verb/GTEA_model/VERB_GTEA_S1_iter_2000.caffemodel GTEA_model/VERB_OBJ_GTEA_S1 4000 3 2>&1 | tee verb_obj_joint_solve_s1.log
python verb_obj_joint_solve.py GTEA_data/S2_train.txt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S2_iter_3000.caffemodel ../verb/GTEA_model/VERB_GTEA_S2_iter_2000.caffemodel GTEA_model/VERB_OBJ_GTEA_S2 4000 3 2>&1 | tee verb_obj_joint_solve_s2.log
python verb_obj_joint_solve.py GTEA_data/S3_train.txt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S3_iter_3000.caffemodel ../verb/GTEA_model/VERB_GTEA_S3_iter_2000.caffemodel GTEA_model/VERB_OBJ_GTEA_S3 4000 3 2>&1 | tee verb_obj_joint_solve_s3.log
python verb_obj_joint_solve.py GTEA_data/S4_train.txt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S4_iter_3000.caffemodel ../verb/GTEA_model/VERB_GTEA_S4_iter_2000.caffemodel GTEA_model/VERB_OBJ_GTEA_S4 4000 3 2>&1 | tee verb_obj_joint_solve_s4.log
