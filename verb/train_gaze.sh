#!/bin/bash
python verb_solve.py verb_train_gaze.prototxt GAZE_data/gaze_verb_train.txt GAZE_model/VERB_GAZE 0.0005 3000 1 2>&1 | tee log/verb_solve_gaze.log
