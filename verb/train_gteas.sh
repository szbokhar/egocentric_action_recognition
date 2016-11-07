#!/bin/bash
python verb_solve.py verb_train_gteas.prototxt GTEAS_data/s1_train.txt GTEAS_model/VERB_GTEAS_S1 0.0005 6000 3 2>&1 | tee log/verb_solve_gteas_s1.log
python verb_solve.py verb_train_gteas.prototxt GTEAS_data/s2_train.txt GTEAS_model/VERB_GTEAS_S2 0.0005 6000 3 2>&1 | tee log/verb_solve_gteas_s2.log
python verb_solve.py verb_train_gteas.prototxt GTEAS_data/s3_train.txt GTEAS_model/VERB_GTEAS_S3 0.0005 6000 3 2>&1 | tee log/verb_solve_gteas_s3.log
python verb_solve.py verb_train_gteas.prototxt GTEAS_data/s4_train.txt GTEAS_model/VERB_GTEAS_S4 0.0005 6000 3 2>&1 | tee log/verb_solve_gteas_s4.log
