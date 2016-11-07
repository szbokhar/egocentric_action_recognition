#!/bin/bash
python verb_solve.py verb_train_plus44m.prototxt PLUS44M_data/ahmad_train.txt PLUS44M_model/VERB_PLUS44M_AHMAD 0.0005 6000 2 2>&1 | tee log/verb_solve_plus44m_ahmad.log
