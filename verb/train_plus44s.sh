#!/bin/bash
#python verb_solve.py verb_train_plus44s.prototxt PLUS44S_data/ahmad_train.txt PLUS44S_model/VERB_PLUS44S_AHMAD 0.0005 6000 0 2>&1 | tee log/verb_solve_plus44s_ahmad.log
python verb_solve.py verb_train_plus44s.prototxt PLUS44S_data/alireza_train.txt PLUS44S_model/VERB_PLUS44S_ALIREZA 0.0005 6000 1 2>&1 | tee log/verb_solve_plus44s_alireza.log
#python verb_solve.py verb_train_plus44s.prototxt PLUS44S_data/carlos_train.txt PLUS44S_model/VERB_PLUS44S_CARLOS 0.0005 6000 2 2>&1 | tee log/verb_solve_plus44s_carlos.log
#python verb_solve.py verb_train_plus44s.prototxt PLUS44S_data/rahul_train.txt PLUS44S_model/VERB_PLUS44S_RAHUL 0.0005 6000 3 2>&1 | tee log/verb_solve_plus44s_rahul.log
