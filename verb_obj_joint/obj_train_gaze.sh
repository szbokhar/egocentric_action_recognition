#!/bin/bash
python obj_solve.py obj_train_gaze.prototxt GAZEO_data/G1_train.txt GAZEO_model/G1 6000 2 2>&1 | tee log/obj_solve_gazeo_g1.log
