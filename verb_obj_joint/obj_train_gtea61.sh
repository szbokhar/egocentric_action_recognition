#!/bin/bash
python obj_solve.py obj_train_gtea61.prototxt GTEA61O_data/S1_train.txt GTEA61O_model/S1 6000 1 2>&1 | tee log/obj_solve_gtea61o_s1.log
python obj_solve.py obj_train_gtea61.prototxt GTEA61O_data/S2_train.txt GTEA61O_model/S2 6000 1 2>&1 | tee log/obj_solve_gtea61o_s2.log
python obj_solve.py obj_train_gtea61.prototxt GTEA61O_data/S3_train.txt GTEA61O_model/S3 6000 1 2>&1 | tee log/obj_solve_gtea61o_s3.log
python obj_solve.py obj_train_gtea61.prototxt GTEA61O_data/S4_train.txt GTEA61O_model/S4 6000 1 2>&1 | tee log/obj_solve_gtea61o_s4.log
