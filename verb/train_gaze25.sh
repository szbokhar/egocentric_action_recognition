#!/bin/bash
python verb_solve.py verb_train_gaze25.prototxt GAZE25_data/gaze_verb_train.txt GAZE25_model/VERB_GAZE25 0.0005 3000 3 2>&1 | tee verb_solve_gaze25.log
