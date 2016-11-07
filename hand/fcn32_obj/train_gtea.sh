#!/bin/bash
#python fcn32_obj_solve.py ../fcn32/GTEA_model/FCN32_GTEA_S1_iter_4000.caffemodel GTEA_data/s1_train.txt GTEA_model/FCN32_OBJ_GTEA_S1 4000 2>&1 | tee fcn32_obj_solve_gtea_s1.log

#python fcn32_obj_solve.py ../fcn32/GTEA_model/FCN32_GTEA_S2_iter_8000.caffemodel GTEA_data/s2_train.txt GTEA_model/FCN32_OBJ_GTEA_S2 4000 2>&1 | tee fcn32_obj_solve_gtea_s2.log

#python fcn32_obj_solve.py ../fcn32/GTEA_model/FCN32_GTEA_S3_iter_4000.caffemodel GTEA_data/s3_train.txt GTEA_model/FCN32_OBJ_GTEA_S3 4000 2>&1 | tee fcn32_obj_solve_gtea_s3.log

python fcn32_obj_solve.py ../fcn32/GTEA_model/FCN32_GTEA_S4_iter_4000.caffemodel GTEA_data/s4_train.txt GTEA_model/FCN32_OBJ_GTEA_S4 4000 2>&1 | tee fcn32_obj_solve_gtea_s4.log
