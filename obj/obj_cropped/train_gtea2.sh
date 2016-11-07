#!/bin/bash
python obj_solve.py obj_train_gtea2.prototxt GTEA_data2/S2_train.txt GTEA_model2/OBJ_GTEA_S2 3000 1 2>&1 | tee log/obj_solve_gtea2_s2.log
