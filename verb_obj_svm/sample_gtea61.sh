python sample_features.py -o ../verb_obj_joint/GTEA61_data/S1_train.txt ../obj/obj_cropped/obj_deploy_gtea61.prototxt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S1_iter_3000.caffemodel GTEA61_features 1 

python sample_features.py -o ../verb_obj_joint/GTEA61_data/S2_train.txt ../obj/obj_cropped/obj_deploy_gtea61.prototxt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S2_iter_3000.caffemodel GTEA61_features 1

python sample_features.py -o ../verb_obj_joint/GTEA61_data/S3_train.txt ../obj/obj_cropped/obj_deploy_gtea61.prototxt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S3_iter_3000.caffemodel GTEA61_features 1

python sample_features.py -o ../verb_obj_joint/GTEA61_data/S4_train.txt ../obj/obj_cropped/obj_deploy_gtea61.prototxt ../obj/obj_cropped/GTEA61_model/OBJ_GTEA61_S4_iter_3000.caffemodel GTEA61_features 1

python sample_features.py ../verb_obj_joint/GTEA61_data/S1_train.txt ../verb/verb_deploy_gtea61.prototxt ../verb/GTEA61_model/VERB_GTEA61_S1_iter_2000.caffemodel GTEA61_features 1

python sample_features.py ../verb_obj_joint/GTEA61_data/S2_train.txt ../verb/verb_deploy_gtea61.prototxt ../verb/GTEA61_model/VERB_GTEA61_S2_iter_2000.caffemodel GTEA61_features 1

python sample_features.py ../verb_obj_joint/GTEA61_data/S3_train.txt ../verb/verb_deploy_gtea61.prototxt ../verb/GTEA61_model/VERB_GTEA61_S3_iter_2000.caffemodel GTEA61_features 1

python sample_features.py ../verb_obj_joint/GTEA61_data/S4_train.txt ../verb/verb_deploy_gtea61.prototxt ../verb/GTEA61_model/VERB_GTEA61_S4_iter_2000.caffemodel GTEA61_features 1


