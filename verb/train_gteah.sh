#!/bin/bash
python verb_solve.py verb_train_gteah.prototxt GTEAH_data/S2_train.txt GTEAH_model/VERB_GTEAH_S2 0.0005 3000 0 2>&1 | tee log/verb_solve_gteah_s2.log
