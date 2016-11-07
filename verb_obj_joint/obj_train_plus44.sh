#!/bin/bash
#python obj_solve.py obj_train_plus44.prototxt PLUS44O_data/AHMAD_train.txt PLUS44O_model/AHMAD 6000 0 2>&1 | tee log/obj_solve_plus44o_ahmad.log
#python obj_solve.py obj_train_plus44.prototxt PLUS44O_data/CARLOS_train.txt PLUS44O_model/CARLOS 6000 0 2>&1 | tee log/obj_solve_plus44o_carlos.log
python obj_solve.py obj_train_plus44.prototxt PLUS44O_data/RAHUL_train.txt PLUS44O_model/RAHUL 6000 1 2>&1 | tee log/obj_solve_plus44o_rahul.log
python obj_solve.py obj_train_plus44.prototxt PLUS44O_data/ALIREZA_train.txt PLUS44O_model/ALIREZA 6000 1 2>&1 | tee log/obj_solve_plus44o_alireza.log
