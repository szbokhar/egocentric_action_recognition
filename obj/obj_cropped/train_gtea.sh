#!/bin/bash
#python obj_solve.py GTEA_data/S1_train.txt GTEA_model/OBJ_GTEA_S1 3000
#python obj_solve.py GTEA_data/S2_train.txt GTEA_model/OBJ_GTEA_S2 3000
#python obj_solve.py GTEA_data/S3_train.txt GTEA_model/OBJ_GTEA_S3 3000 2>&1 | tee obj_solve_gtea_s3.log
python obj_solve.py GTEA_data/S4_train.txt GTEA_model/OBJ_GTEA_S4 3000 2>&1 | tee obj_solve_gtea_s4.log
