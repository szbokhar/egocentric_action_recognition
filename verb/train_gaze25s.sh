#!/bin/bash
python verb_solve.py verb_train_gaze25s.prototxt GAZE25S_data/G1_train.txt GAZE25S_model/VERB_GAZE25S 0.0005 3000 2 2>&1 | tee log/verb_solve_gaze25s.log
