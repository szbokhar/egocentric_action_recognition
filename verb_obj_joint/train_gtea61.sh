#!/bin/bash
#python verb_obj_joint_solve.py verb_obj_joint_train_gtea61.prototxt ../verb/verb_deploy_gtea61.prototxt ../obj/obj_cropped/obj_deploy_gtea61.prototxt GTEA61_data/S1_train.txt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S1_iter_3000.caffemodel ../verb/GTEA61_model/VERB_GTEA61_S1_iter_2000.caffemodel GTEA61_model/VERB_OBJ_GTEA61_S1 4000 3 2>&1 | tee log/verb_obj_joint_solve_gtea61_s1.log

python verb_obj_joint_solve.py verb_obj_joint_train_gtea61.prototxt ../verb/verb_deploy_gtea61.prototxt ../obj/obj_cropped/obj_deploy_gtea61.prototxt GTEA61_data/S2_train.txt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S2_iter_3000.caffemodel ../verb/GTEA61_model/VERB_GTEA61_S2_iter_2000.caffemodel GTEA61_model/VERB_OBJ_GTEA61_S2 4000 3 2>&1 | tee log/verb_obj_joint_solve_gtea61_s2.log

python verb_obj_joint_solve.py verb_obj_joint_train_gtea61.prototxt ../verb/verb_deploy_gtea61.prototxt ../obj/obj_cropped/obj_deploy_gtea61.prototxt GTEA61_data/S3_train.txt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S3_iter_3000.caffemodel ../verb/GTEA61_model/VERB_GTEA61_S3_iter_2000.caffemodel GTEA61_model/VERB_OBJ_GTEA61_S3 4000 3 2>&1 | tee log/verb_obj_joint_solve_gtea61_s3.log

python verb_obj_joint_solve.py verb_obj_joint_train_gtea61.prototxt ../verb/verb_deploy_gtea61.prototxt ../obj/obj_cropped/obj_deploy_gtea61.prototxt GTEA61_data/S4_train.txt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S4_iter_3000.caffemodel ../verb/GTEA61_model/VERB_GTEA61_S4_iter_2000.caffemodel GTEA61_model/VERB_OBJ_GTEA61_S4 4000 3 2>&1 | tee log/verb_obj_joint_solve_gtea61_s4.log
