#!/bin/bash
python fcn32_hands_solve.py ../fcn32/GTEA_model/FCN32_GTEA_S2_iter_8000.caffemodel GTEA_data/s2_train.txt GTEA_model/FCN32_HANDS_GTEA_S2 8000 2>&1 | tee fcn32_hands_solve_gtea_s2.log 
