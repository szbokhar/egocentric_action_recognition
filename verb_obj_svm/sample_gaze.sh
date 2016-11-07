#python sample_features.py -o ../verb_obj_joint/GAZE_data/G1_train.txt ../obj/obj_cropped/obj_deploy_gaze.prototxt ../obj/obj_cropped/GAZE_model/OBJ_GAZE_iter_1000.caffemodel GAZE_features 1

#python sample_features.py -o ../verb_obj_joint/GAZE_data/G1_test.txt ../obj/obj_cropped/obj_deploy_gaze.prototxt ../obj/obj_cropped/GAZE_model/OBJ_GAZE_iter_1000.caffemodel GAZE_features 1

python sample_features2.py ../verb_obj_joint/GAZE_data/G1_train.txt ../verb/verb_deploy_gaze.prototxt ../verb/GAZE_model/VERB_GAZE_iter_1000.caffemodel GAZE_features 1


python sample_features2.py ../verb_obj_joint/GAZE_data/G1_test.txt ../verb/verb_deploy_gaze.prototxt ../verb/GAZE_model/VERB_GAZE_iter_1000.caffemodel GAZE_features 1

