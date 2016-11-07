python sample_features.py -o GAZE25_data/G1_test.txt ../obj/obj_raw/obj_deploy_gaze25.prototxt ../obj/obj_raw/GAZE25_model/G1_iter_1000.caffemodel GAZE25_features 2
python sample_features.py -o GAZE25_data/G1_train.txt ../obj/obj_raw/obj_deploy_gaze25.prototxt ../obj/obj_raw/GAZE25_model/G1_iter_1000.caffemodel GAZE25_features 2
python sample_features.py GAZE25_data/G1_test.txt ../verb/verb_deploy_gaze25s.prototxt ../verb/GAZE25S_model/VERB_GAZE25S_iter_1000.caffemodel GAZE25_features 2
python sample_features.py GAZE25_data/G1_train.txt ../verb/verb_deploy_gaze25s.prototxt ../verb/GAZE25S_model/VERB_GAZE25S_iter_1000.caffemodel GAZE25_features 2


