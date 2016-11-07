#!/bin/bash
python verb_solve.py verb_train_plus44.prototxt PLUS44H20B_data/AHMAD_train.txt PLUS44H20B_model/VERB_PLUS44H20B_AHMAD 0.0005 3000 3 2>&1 | tee log/verb_solve_plus44h20b_ahmad.log
#python verb_solve.py verb_train_plus44.prototxt PLUS44H_data/alireza_train.txt PLUS44H_model/VERB_PLUS44H_ALIREZA 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44h_alireza.log
#python verb_solve.py verb_train_plus44.prototxt PLUS44H_data/carlos_train.txt PLUS44H_model/VERB_PLUS44H_CARLOS 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44h_carlos.log
#python verb_solve.py verb_train_plus44.prototxt PLUS44H_data/rahul_train.txt PLUS44H_model/VERB_PLUS44H_RAHUL 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44h_rahul.log
