#!/bin/bash
#python obj_solve.py obj_train_gtea.prototxt GTEAO_data/S1_train.txt GTEAO_model/S1 6000 2 2>&1 | tee log/obj_solve_gteao_s1.log
#python obj_solve.py obj_train_gtea.prototxt GTEAO_data/S2_train.txt GTEAO_model/S2 6000 2 2>&1 | tee log/obj_solve_gteao_s2.log
#python obj_solve.py obj_train_gtea.prototxt GTEAO_data/S3_train.txt GTEAO_model/S3 6000 3 2>&1 | tee log/obj_solve_gteao_s3.log
python obj_solve.py obj_train_gtea.prototxt GTEAO_data/S4_train.txt GTEAO_model/S4 6000 3 2>&1 | tee log/obj_solve_gteao_s4.log
