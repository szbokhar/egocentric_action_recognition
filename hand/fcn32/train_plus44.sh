#!/bin/bash
#python fcn32_solve.py PLUS44_data/ahmad_train.txt PLUS44_model/FCN32_PLUS44_AHMAD 8000 1 2>&1 | tee log/solve_plus44_ahmad.log
python fcn32_solve.py PLUS44_data/carlos_train.txt PLUS44_model/FCN32_PLUS44_CARLOS 8000 3 2>&1 | tee log/solve_plus44_carlos.log
python fcn32_solve.py PLUS44_data/alireza_train.txt PLUS44_model/FCN32_PLUS44_ALIREZA 8000 3 2>&1 | tee log/solve_plus44_alireza.log
python fcn32_solve.py PLUS44_data/rahul_train.txt PLUS44_model/FCN32_PLUS44_RAHUL 8000 3 2>&1 | tee log/solve_plus44_rahul.log
