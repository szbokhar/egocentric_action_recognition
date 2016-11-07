#!/bin/bash
python obj_solve.py obj_train_plusm.prototxt PLUSM_data/AHMAD_train.txt PLUSM_model/OBJ_PLUSM_AHMAD 3000 0 2>&1 | tee log/obj_solve_plusm_ahmad.log
