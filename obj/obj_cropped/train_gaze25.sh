#!/bin/bash
python obj_solve.py obj_train_gaze25.prototxt GAZE25_data/gaze_obj_train.txt GAZE25_model/OBJ_GAZE25 3000 3 2>&1 | tee obj_solve_gaze25.log
