#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

# Note: (1) To reproduce published results, use the models in "Trained_Models"
#       (2) Uncomment the sections of code you wish to run
#       (3) Run *.m files in MATLAB

######################################### DATASET PREPARATION ################################################

# Split datasets:
# Segmentation_splitDatasets.m

############################################### LIMBUS-NET ##################################################

# Directories:
MAIN_DIR='../../Models/Segmentation/Limbus-Net/Blue_Light'
DATASET_DIR='../../Datasets/Segmentation/Blue_Light'

# Training and validation:
# for K_TAG in {'K1','K2','K3','K4','K5','K6'}
# do
#   python LimbusNet_train_blue.py "$MAIN_DIR" $K_TAG 300 1 300 0 0 "$DATASET_DIR"
#   python LimbusNet_validate_blue.py "$MAIN_DIR" $K_TAG 300 "$DATASET_DIR"
# done

# Testing (replace xx with threshold from validation/average_dsc_metric.png):
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K1' 300 xx "$DATASET_DIR"
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K2' 300 xx "$DATASET_DIR"
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K3' 300 xx "$DATASET_DIR"
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K4' 300 xx "$DATASET_DIR"
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K5' 300 xx "$DATASET_DIR"
# python LimbusNet_test_blue.py "$MAIN_DIR" 'K6' 300 xx "$DATASET_DIR"

# Update dataset with Limbus-Net predictions:
# LimbusNet_updateDatasetWithPredictions.m

############################################## SLIT-NET V2 ##################################################

# Directories:
MAIN_DIR='../../Models/Segmentation/SLIT-Net_v2/Blue_Light'
DATASET_DIR='../../Datasets/Segmentation/Blue_Light'

# Training and validation:
# for K_TAG in {'K1','K2','K3','K4','K5','K6'}
# do
#   python SLITNet_v2_train_blue.py "$MAIN_DIR" $K_TAG 1 'coco' 0 300 $K_TAG 0 299 1 "$DATASET_DIR"
#   python SLITNet_v2_validateThresholdClass_blue.py "$MAIN_DIR" $K_TAG 299 $K_TAG "$DATASET_DIR"
#   python SLITNet_v2_validateThresholdMask_blue.py "$MAIN_DIR" $K_TAG 299 $K_TAG "$DATASET_DIR"
# done

# Testing (including using Limbus-Net predictions):
# python SLITNet_v2_test_blue.py "$MAIN_DIR" 299 "$DATASET_DIR"
