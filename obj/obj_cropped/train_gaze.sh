#!/bin/bash
python obj_solve.py obj_train_gaze.prototxt GAZE_data/gaze_obj_train.txt GAZE_model/OBJ_GAZE 3000 1 2>&1 | tee log/obj_solve_gaze.log
