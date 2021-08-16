#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Note: (1) To reproduce published results, use the models in "Trained_Models"
#       (2) Uncomment the sections of code you wish to run
#       (3) Run *.m files in MATLAB

######################################### DATASET PREPARATION ###############################################

# Create and split datasets:
# BCVA_Prediction_createDataset.m
# BCVA_Prediction_updateDatasetWithMeasurements.m
# BCVA_Prediction_splitDatasets.m

############################################# BCVA PREDICTION ###############################################

# Directories:
MAIN_DIR='../../Models/BCVA_Prediction'
DATASET_DIR='../../Datasets/BCVA_Prediction'

# Training:
# python BCVA_Prediction_train.py "$MAIN_DIR" "$DATASET_DIR"

# Testing:
# python BCVA_Prediction_test.py "$MAIN_DIR" "$DATASET_DIR"
