#!/bin/bash
python obj_solve.py GTEA61_data/S1_train.txt GTEA61_model/OBJ_GTEA61_S1 3000 2 2>&1 | tee obj_solve_gtea61_s1.log
python obj_solve.py GTEA61_data/S2_train.txt GTEA61_model/OBJ_GTEA61_S2 3000 2 2>&1 | tee obj_solve_gtea61_s2.log
python obj_solve.py GTEA61_data/S3_train.txt GTEA61_model/OBJ_GTEA61_S3 3000 2 2>&1 | tee obj_solve_gtea61_s3.log
python obj_solve.py GTEA61_data/S4_train.txt GTEA61_model/OBJ_GTEA61_S4 3000 2 2>&1 | tee obj_solve_gtea61_s4.log
