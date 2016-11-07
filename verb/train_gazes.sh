#!/bin/bash
python verb_solve.py verb_train_gazes.prototxt GAZES_data/G1_train.txt GAZES_model/VERB_GAZES 0.0005 3000 1 2>&1 | tee log/verb_solve_gazes.log
