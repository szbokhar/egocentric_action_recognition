#!/bin/bash
python verb_solve.py GTEA61_data/S1_train.txt GTEA61_model/VERB_GTEA61_S1 4000 2 2>&1 | tee verb_solve_gtea61_s1.log
#python verb_solve.py GTEA61_data/S2_train.txt GTEA61_model/VERB_GTEA61_S2 2000 3 2>&1 | tee verb_solve_gtea61_s2.log
#python verb_solve.py GTEA61_data/S3_train.txt GTEA61_model/VERB_GTEA61_S3 2000 3 2>&1 | tee verb_solve_gtea61_s3.log
#python verb_solve.py GTEA61_data/S4_train.txt GTEA61_model/VERB_GTEA61_S4 2000 3 2>&1 | tee verb_solve_gtea61_s4.log
