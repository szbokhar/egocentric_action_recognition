python sample_features.py -o GTEA_data/S1_test.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S1_iter_2000.caffemodel GTEA_features 1
python sample_features.py -o GTEA_data/S1_train.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S1_iter_2000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S1_test.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S1_iter_4000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S1_train.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S1_iter_4000.caffemodel GTEA_features 1

python sample_features.py -o GTEA_data/S2_test.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S2_iter_2000.caffemodel GTEA_features 1
python sample_features.py -o GTEA_data/S2_train.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S2_iter_2000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S2_test.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S2_iter_4000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S2_train.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S2_iter_4000.caffemodel GTEA_features 1

python sample_features.py -o GTEA_data/S3_test.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S3_iter_2000.caffemodel GTEA_features 1
python sample_features.py -o GTEA_data/S3_train.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S3_iter_2000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S3_test.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S3_iter_4000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S3_train.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S3_iter_4000.caffemodel GTEA_features 1

python sample_features.py -o GTEA_data/S4_test.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S4_iter_2000.caffemodel GTEA_features 1
python sample_features.py -o GTEA_data/S4_train.txt ../obj/obj_raw/obj_deploy_gteas.prototxt ../obj/obj_raw/GTEAS_model/S4_iter_2000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S4_test.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S4_iter_4000.caffemodel GTEA_features 1
python sample_features.py GTEA_data/S4_train.txt ../verb/verb_deploy_gteas.prototxt ../verb/GTEAS_model/VERB_GTEAS_S4_iter_4000.caffemodel GTEA_features 1


