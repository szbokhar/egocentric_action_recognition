python sample_features.py -o ../verb_obj_joint/GTEA_data/S1_train.txt ../obj/obj_cropped/obj_deploy_gtea.prototxt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S1_iter_3000.caffemodel GTEA_features 2

python sample_features.py -o ../verb_obj_joint/GTEA_data/S2_train.txt ../obj/obj_cropped/obj_deploy_gtea.prototxt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S2_iter_3000.caffemodel GTEA_features 1

python sample_features.py -o ../verb_obj_joint/GTEA_data/S3_train.txt ../obj/obj_cropped/obj_deploy_gtea.prototxt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S3_iter_3000.caffemodel GTEA_features 0

python sample_features.py -o ../verb_obj_joint/GTEA_data/S4_train.txt ../obj/obj_cropped/obj_deploy_gtea.prototxt ../obj/obj_cropped/GTEA_model/OBJ_GTEA_S4_iter_3000.caffemodel GTEA_features 0

python sample_features.py ../verb_obj_joint/GTEA_data/S1_train.txt ../verb/verb_deploy_gtea.prototxt ../verb/GTEA_model/VERB_GTEA_S1_iter_2000.caffemodel GTEA_features 2

python sample_features.py ../verb_obj_joint/GTEA_data/S2_train.txt ../verb/verb_deploy_gtea.prototxt ../verb/GTEA_model/VERB_GTEA_S2_iter_2000.caffemodel GTEA_features 1

python sample_features.py ../verb_obj_joint/GTEA_data/S3_train.txt ../verb/verb_deploy_gtea.prototxt ../verb/GTEA_model/VERB_GTEA_S3_iter_2000.caffemodel GTEA_features 0

python sample_features.py ../verb_obj_joint/GTEA_data/S4_train.txt ../verb/verb_deploy_gtea.prototxt ../verb/GTEA_model/VERB_GTEA_S4_iter_2000.caffemodel GTEA_features 0


