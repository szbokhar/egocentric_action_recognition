python sample_features.py -o GAZE_data/G1_test.txt ../obj/obj_raw/obj_deploy_gaze.prototxt ../obj/obj_raw/GAZE_model/G1_iter_1000.caffemodel GAZE_features 1
python sample_features.py -o GAZE_data/G1_train.txt ../obj/obj_raw/obj_deploy_gaze.prototxt ../obj/obj_raw/GAZE_model/G1_iter_1000.caffemodel GAZE_features 1
python sample_features.py GAZE_data/G1_test.txt ../verb/verb_deploy_gazes.prototxt ../verb/GAZES_model/VERB_GAZES_iter_2000.caffemodel GAZE_features 1
python sample_features.py GAZE_data/G1_train.txt ../verb/verb_deploy_gazes.prototxt ../verb/GAZES_model/VERB_GAZES_iter_2000.caffemodel GAZE_features 1


