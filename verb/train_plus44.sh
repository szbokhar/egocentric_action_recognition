#!/bin/bash
#python verb_solve.py verb_train_plus44.prototxt PLUS44_data/ahmad_train.txt PLUS44_model/VERB_PLUS44_AHMAD 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44_ahmad.log
#python verb_solve.py verb_train_plus44.prototxt PLUS44_data/alireza_train.txt PLUS44_model/VERB_PLUS44_ALIREZA 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44_alireza.log
#python verb_solve.py verb_train_plus44.prototxt PLUS44_data/carlos_train.txt PLUS44_model/VERB_PLUS44_CARLOS 0.0005 3000 2 2>&1 | tee log/verb_solve_plus44_carlos.log
python verb_solve.py verb_train_plus44.prototxt PLUS44_data/rahul_train.txt PLUS44_model/VERB_PLUS44_RAHUL 0.0005 3000 3 2>&1 | tee log/verb_solve_plus44_rahul.log
