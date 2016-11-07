#!/bin/bash
#python obj_solve.py obj_train_plus44.prototxt PLUS44_data/AHMAD_train.txt PLUS44_model/OBJ_PLUS44_AHMAD 3000 0 2>&1 | tee log/obj_solve_plus44_ahmad.log
python obj_solve.py obj_train_plus44.prototxt PLUS44_data/CARLOS_train.txt PLUS44_model/OBJ_PLUS44_CARLOS 3000 2 2>&1 | tee log/obj_solve_plus44_carlos.log
python obj_solve.py obj_train_plus44.prototxt PLUS44_data/RAHUL_train.txt PLUS44_model/OBJ_PLUS44_RAHUL 3000 2 2>&1 | tee log/obj_solve_plus44_rahul.log
python obj_solve.py obj_train_plus44.prototxt PLUS44_data/ALIREZA_train.txt PLUS44_model/OBJ_PLUS44_ALIREZA 3000 2 2>&1 | tee log/obj_solve_plus44_alireza.log
