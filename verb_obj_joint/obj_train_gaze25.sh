#!/bin/bash
python obj_solve.py obj_train_gaze25.prototxt GAZE25O_data/G1_train.txt GAZE25O_model/G1 6000 2 2>&1 | tee log/obj_solve_gaze25o_g1.log
