#!/bin/bash

# For flow_mask
rm flow_mask/GTEA_{data,model,output}
rm flow_mask/PLUS44_{data,model,output}

# For hand/fcn32
rm hand/fcn32/GTEA_{data,model,output}
rm hand/fcn32/PLUS44_{data,model,output}
rm hand/fcn32/log

# For hand/fcn32_hands
rm hand/fcn32_hands/GTEA_{data,model,output}

# For hand/fcn32_hands_obj
rm hand/fcn32_hands_obj/GTEA_data

# For hand/fcn32_obj
rm hand/fcn32_obj/GTEA_{data,model,output,output2}
rm hand/fcn32_obj/PLUS44_{output,output2,output_mini,output_mini2}
rm hand/fcn32_obj/PLUS_{data,model,output,prob_output}
rm hand/fcn32_obj/PLUSM_{output,output_mini}

# For detection/GTEA
rm detection/GTEA/data

# For detection/PLUS44
rm detection/PLUS44/data

# For temporal
rm temporal/GTEA_data
rm temporal/GTEA_model

# For decouple_verb_obj_joint
rm decouple_verb_obj_joint/GTEA_{data,model,output}
rm decouple_verb_obj_joint/log
rm decouple_verb_obj_joint/PLUS44_{data,model,model15,output}

# For obj/obj_cropped
rm obj/obj_cropped/GAZE25_{data,model}
rm obj/obj_cropped/GAZE_{data,model,output}
rm obj/obj_cropped/GTEA61_{data,model,output}
rm obj/obj_cropped/GTEA_{data,model,output}
rm obj/obj_cropped/GTEA_{data2,model2,output2}
rm obj/obj_cropped/log
rm obj/obj_cropped/PLUS44_{data,model,output}
rm obj/obj_cropped/PLUS44_{data2,model2,output2}
rm obj/obj_cropped/PLUSM_{data,model}

# For obj/obj_raw
rm obj/obj_raw/GAZE25_{data,model,output}
rm obj/obj_raw/GAZE_{data,model,output}
rm obj/obj_raw/GTEA61_{data,model,output}
rm obj/obj_raw/GTEA_{data,model}
rm obj/obj_raw/GTEAS_{data,model,output}
rm obj/obj_raw/PLUS44_{data,model,output}
rm obj/obj_raw/PLUS44S_{data,model,output}

# For verb_obj_hand_joint
rm verb_obj_hand_joint/GTEA_{data,model,output}
rm verb_obj_hand_joint/PLUS44_{data,model,output}

# For 2stream_joint
rm 2stream_joint/GAZE25_{data,features,model,output}
rm 2stream_joint/GAZE_{data,features,model,output}
rm 2stream_joint/GTEA61_{data,features,model,output}
rm 2stream_joint/GTEA_{data,features,model,output}
rm 2stream_joint/PLUS44_{data,features,model,output}

# For spatial
rm spatial/GTEA_{data,model}

# For verb_hand
rm verb_hand/GTEAH_{data,model,model_bak,output}
rm verb_hand/PLUS44_{data,model,model00,model05,model10,model15,output}

# For fuse_svm
rm fuse_svm/GTEA_{data,model,output}
rm fuse_svm/log
rm fuse_svm/PLUS44_{data,features,model,output}
rm fuse_svm/SVM_model

# For verb_obj_svm
rm verb_obj_svm/GAZE25_features
rm verb_obj_svm/GAZE_features
rm verb_obj_svm/GTEA61_features
rm verb_obj_svm/GTEA_features
rm verb_obj_svm/PLUS44_features
rm verb_obj_svm/SVM
