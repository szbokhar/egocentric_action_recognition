#!/bin/bash
python verb_solve.py verb_train_gtea61s.prototxt GTEA61S_data/s1_train.txt GTEA61S_model/VERB_GTEA61S_S1 0.0005 6000 0 2>&1 | tee log/verb_solve_gtea61s_s1.log
python verb_solve.py verb_train_gtea61s.prototxt GTEA61S_data/s2_train.txt GTEA61S_model/VERB_GTEA61S_S2 0.0005 6000 0 2>&1 | tee log/verb_solve_gtea61s_s2.log
python verb_solve.py verb_train_gtea61s.prototxt GTEA61S_data/s3_train.txt GTEA61S_model/VERB_GTEA61S_S3 0.0005 6000 0 2>&1 | tee log/verb_solve_gtea61s_s3.log
python verb_solve.py verb_train_gtea61s.prototxt GTEA61S_data/s4_train.txt GTEA61S_model/VERB_GTEA61S_S4 0.0005 6000 0 2>&1 | tee log/verb_solve_gtea61s_s4.log
