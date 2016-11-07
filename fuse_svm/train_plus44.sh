#!/bin/bash
python obj_solve.py obj_train_plus44.prototxt PLUS44_data/AHMAD_train.txt PLUS44_model/OBJ_PLUS44_AHMAD 3000 0 2>&1 | tee log/obj_solve_plus44_ahmad.log
