#python sample_features.py -o GTEA61_data/S1_test.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S1_iter_2000.caffemodel GTEA61_features 1
#python sample_features.py -o GTEA61_data/S1_train.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S1_iter_2000.caffemodel GTEA61_features 1
#python sample_features.py GTEA61_data/S1_test.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S1_iter_4000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S1_train.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S1_iter_4000.caffemodel GTEA61_features 1

python sample_features.py -o GTEA61_data/S2_test.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S2_iter_2000.caffemodel GTEA61_features 1
python sample_features.py -o GTEA61_data/S2_train.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S2_iter_2000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S2_test.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S2_iter_4000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S2_train.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S2_iter_4000.caffemodel GTEA61_features 1

python sample_features.py -o GTEA61_data/S3_test.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S3_iter_2000.caffemodel GTEA61_features 1
python sample_features.py -o GTEA61_data/S3_train.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S3_iter_2000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S3_test.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S3_iter_4000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S3_train.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S3_iter_4000.caffemodel GTEA61_features 1

python sample_features.py -o GTEA61_data/S4_test.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S4_iter_2000.caffemodel GTEA61_features 1
python sample_features.py -o GTEA61_data/S4_train.txt ../obj/obj_raw/obj_deploy_gtea61.prototxt ../obj/obj_raw/GTEA61_model/S4_iter_2000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S4_test.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S4_iter_4000.caffemodel GTEA61_features 1
python sample_features.py GTEA61_data/S4_train.txt ../verb/verb_deploy_gtea61s.prototxt ../verb/GTEA61S_model/VERB_GTEA61S_S4_iter_4000.caffemodel GTEA61_features 1


