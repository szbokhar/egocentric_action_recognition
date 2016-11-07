#!/bin/bash
python verb_solve.py GTEA_data/S1_train.txt GTEA_model/VERB_GTEA_S1 2000 0 2>&1 | tee verb_solve_gtea_s1.log
#python verb_solve.py GTEA_data/S2_train.txt GTEA_model/VERB_GTEA_S2 2000 2>&1 | tee verb_solve_gtea_s2.log
#python verb_solve.py GTEA_data/S3_train.txt GTEA_model/VERB_GTEA_S3 2000 2>&1 | tee verb_solve_gtea_s3.log
#python verb_solve.py GTEA_data/S4_train.txt GTEA_model/VERB_GTEA_S4 2000 2>&1 | tee verb_solve_gtea_s4.log
