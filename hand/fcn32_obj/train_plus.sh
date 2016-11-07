#!/bin/bash
#python fcn32_obj_solve.py ../fcn32/PLUS44_model/FCN32_PLUS44_AHMAD_iter_4000.caffemodel PLUS_data/ahmad_train.txt PLUS_model/FCN32_OBJ_PLUS_AHMAD 4000 1 2>&1 | tee log/fcn32_obj_solve_plus_ahmad.log
python fcn32_obj_solve.py ../fcn32/PLUS44_model/FCN32_PLUS44_AHMAD_iter_4000.caffemodel PLUS_data/alireza_train.txt PLUS_model/FCN32_OBJ_PLUS_ALIREZA 4000 1 2>&1 | tee log/fcn32_obj_solve_plus_alireza.log
python fcn32_obj_solve.py ../fcn32/PLUS44_model/FCN32_PLUS44_AHMAD_iter_4000.caffemodel PLUS_data/carlos_train.txt PLUS_model/FCN32_OBJ_PLUS_CARLOS 4000 1 2>&1 | tee log/fcn32_obj_solve_plus_carlos.log
python fcn32_obj_solve.py ../fcn32/PLUS44_model/FCN32_PLUS44_AHMAD_iter_4000.caffemodel PLUS_data/rahul_train.txt PLUS_model/FCN32_OBJ_PLUS_RAHUL 4000 1 2>&1 | tee log/fcn32_obj_solve_plus_rahul.log
