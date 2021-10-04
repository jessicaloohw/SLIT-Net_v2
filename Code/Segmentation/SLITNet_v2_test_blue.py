# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys
import pickle
import time
import warnings
import numpy as np
import h5py as hh
import scipy.io as sio

import SLITNet_v2_model as modellib
from SLITNet_v2_utils import apply_class_thresholds
from SLITNet_v2_utils import get_predictions_in_original_size as get_predictions
from SLITNet_v2_utils import restrict_within_limbus
from SLITNet_v2_utils import clean_predictions_blue as clean_predictions
from SLITNet_v2_utils import calculate_performance_metrics
from SLITNet_v2_utils import get_limbus_length
from SLITNet_v2_utils import display_instances_ulcer_colorscheme

from SLITNet_v2_utils import convert_image_for_testing
from SLITNet_v2_utils import load_image_and_annotations_with_limbus_net as load_image_and_annotations
from SLITNet_v2_utils import convert_results_to_full_size_with_limbus_net as convert_results_to_full_size


# Settings:
MODE = "inference"


def main():

    ################################# USER INPUT #################################

    # System inputs:
    MODEL_DIR = sys.argv[1]
    MODEL_NUM = int(sys.argv[2])
    DATASET_DIR = sys.argv[3]

    #########################################################################3####

    # Thresholds:
    THRESHOLD_CLASS = 0.0
    THRESHOLD_NMS = 0.5

    # Rules:
    RULES = []

    ##############################################################################

    # Config:
    CONFIG_FILENAME = os.path.join(MODEL_DIR, 'config.pickle')
    config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
    if (MODE == "inference"):
        config.set_to_inference_mode(class_threshold=THRESHOLD_CLASS, nms_threshold=THRESHOLD_NMS)
    config.display()

    # Number of classes:
    NUM_CLASSES = config.NUM_CLASSES

    # Main save directory:
    MAIN_SAVE_DIR = os.path.join(MODEL_DIR, 'Final_Segmentations')
    if not os.path.exists(MAIN_SAVE_DIR):
        os.makedirs(MAIN_SAVE_DIR)

    #################################################################################

    # Initialize file:
    def initialize_testing_metrics(metrics_dir, metrics_filename, num_classes):
        filename = os.path.join(metrics_dir, '{}.txt'.format(metrics_filename))
        with open(filename, 'w') as wf:
            wf.write('\t\t\tDSC')
            for i in range(num_classes - 1):
                wf.write('\t')
            wf.write('\tLimbus')
            wf.write('\nK-fold\tImage\tVXXX\t')
            for i in range(num_classes):
                wf.write('Class_{}\t'.format(i + 1))
            wf.write('Truth_Length\tPred_Length')
        return filename

    WRITE_FILENAME = initialize_testing_metrics(MAIN_SAVE_DIR, 'testing', NUM_CLASSES)

    ###############################################################################

    for K_FOLD in ['K1','K2','K3','K4','K5','K6']:

        # Model sub-directory:
        MODEL_SUBDIR = K_FOLD

        # Check existence:
        if not os.path.exists(os.path.join(MODEL_DIR, MODEL_SUBDIR)):
            print('({}) The folder does not exist.'.format(K_FOLD))
            continue

        # Best thresholds:
        BEST_THRESHOLD_FILE = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'validation', 'model-{}'.format(MODEL_NUM),
                                           'THRESHOLD_CLASS_best_thresholds.npy')
        if os.path.exists(BEST_THRESHOLD_FILE):
            THRESHOLDS = np.load(BEST_THRESHOLD_FILE)
        else:
            print('({}) THRESHOLDS not found in file.'.format(K_FOLD))
            continue

        BEST_THRESHOLD_MASK_FILE = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'validation', 'model-{}'.format(MODEL_NUM),
                                                'THRESHOLD_MASK_best_thresholds.npy')
        if os.path.exists(BEST_THRESHOLD_MASK_FILE):
            THRESHOLD_MASK = np.load(BEST_THRESHOLD_MASK_FILE)
        else:
            print('({}) THRESHOLD_MASK not found in file.'.format(K_FOLD))
            continue

        ########################################################################################

        # Dataset:
        TEST_FILENAME = os.path.join(DATASET_DIR, 'test_data_{}.mat'.format(K_FOLD))
        all_images, all_ground_truth, all_limbus_net_mask, all_limbus_net_box, all_vxxx = load_image_and_annotations(TEST_FILENAME)
        NUM_IMAGES = len(all_images)
        print('Test dataset: {}'.format(TEST_FILENAME))
        print('Test dataset prepared: {} images.'.format(NUM_IMAGES))

        # Ignore these specific warnings:
        warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15 to")
        warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")

        # Recreate the model in inference mode:
        model = modellib.MaskRCNN(mode=MODE,
                                  config=config,
                                  model_dir=os.path.join(MODEL_DIR, MODEL_SUBDIR))
        model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'mask_rcnn_ulcer_' + str(MODEL_NUM).zfill(4) + '.h5')

        # Load weights:
        model.load_weights(model_path,
                           by_name=True)
        print('Loading weights from: {}'.format(model_path))

        # Save images:
        for image_id in range(NUM_IMAGES):

            # Get VXXX:
            VXXX = all_vxxx[image_id]

            # Create save directory:
            SAVE_DIR = os.path.join(MAIN_SAVE_DIR, VXXX)
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            else:
                print('({}) {} folder exists and files will be overwritten. Please check.'.format(K_FOLD, VXXX))

            # Load image, manual annotations, and Limbus-Net predictions:
            image = all_images[image_id]
            ground_truth = all_ground_truth[image_id]
            limbus_net_mask = all_limbus_net_mask[image_id]
            limbus_net_box = all_limbus_net_box[image_id]

            # Convert to image_for_testing for model detection:
            image_for_testing = image[limbus_net_box[0, 0]:limbus_net_box[2, 0], limbus_net_box[1, 0]:limbus_net_box[3, 0]]
            image_for_testing, image_meta = convert_image_for_testing(config, image_id, image_for_testing)

            # Run model detection:
            start_time = time.time()
            results_from_model = model.detect([image_for_testing])[0]
            elapsed_time = time.time() - start_time
            print('({}) Image {}: {} seconds'.format(K_FOLD, image_id, elapsed_time))

            # Apply thresholds and get predictions:
            results = apply_class_thresholds(results_from_model, THRESHOLDS)
            results = get_predictions(image_meta, results, THRESHOLD_MASK)
            results = convert_results_to_full_size(results, limbus_net_box, image.shape[0], image.shape[1])

            # Add Limbus-Net predictions to results for convenience:
            results["class_ids"] = np.concatenate((results["class_ids"], [4]))
            results["rois"] = np.concatenate((results["rois"], np.transpose(limbus_net_box)), axis=0)
            results["masks"] = np.concatenate((results["masks"], limbus_net_mask[:, :, None]), axis=2)
            results["scores"] = np.concatenate((results["scores"], [1.0]))

            # Clean predictions:
            results = restrict_within_limbus(results, IDS_TO_RESTRICT=[1, 3], HIGH_OVERLAP=0.7, LIMBUS_ID=4)
            results = clean_predictions(results, RULES)

            # Save segmentations as MAT file:
            sio.savemat(os.path.join(SAVE_DIR, 'segmentations.mat'), results)

            ################################### DISPLAY ###########################################

            # Ground truth:
            display_instances_ulcer_colorscheme(image=image,
                                                boxes=ground_truth["rois"],
                                                masks=ground_truth["masks"],
                                                class_ids=ground_truth["class_ids"],
                                                title='{} | manual'.format(VXXX),
                                                show_mask=False,
                                                show_bbox=False,
                                                blue_light=True)
            save_filename = os.path.join(SAVE_DIR, 'manual.png')
            plt.savefig(save_filename)
            plt.close()

            # Prediction:
            display_instances_ulcer_colorscheme(image=image,
                                                boxes=results["rois"],
                                                masks=results["masks"],
                                                class_ids=results["class_ids"],
                                                title='{} | auto'.format(VXXX),
                                                show_mask=False,
                                                show_bbox=False,
                                                blue_light=True)
            save_filename = os.path.join(SAVE_DIR, 'auto.png')
            plt.savefig(save_filename)
            plt.close()

            ################################### METRICS ###############################################

            def write_testing_metrics(write_filename, num_classes,
                                      dsc_metrics, truth_limbus_length, pred_limbus_length):
                with open(write_filename, 'a') as wf:
                    wf.write('\n{}\t{}\t{}\t'.format(K_FOLD, image_id, VXXX))
                    for i in range(num_classes):
                        wf.write('{}\t'.format(dsc_metrics[i, 0]))
                    wf.write('{}\t{}'.format(truth_limbus_length, pred_limbus_length))

            # DSC:
            class_dsc, _, _ = calculate_performance_metrics(pred=results,
                                                            truth=ground_truth,
                                                            num_classes=NUM_CLASSES)

            # Limbus length:
            truth_limbus_length = get_limbus_length(ground_truth, limbus_id=4)
            pred_limbus_length = get_limbus_length(results, limbus_id=4)

            # Write to file:
            write_testing_metrics(WRITE_FILENAME, NUM_CLASSES,
                                  class_dsc, truth_limbus_length, pred_limbus_length)

    print('Finished.')


if __name__ == '__main__':
    main()
