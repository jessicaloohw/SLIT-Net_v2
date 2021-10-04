# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys
import pickle
import warnings
import numpy as np

from mrcnn.utils import resize_image, resize_mask

import SLITNet_v2_model as modellib
from SLITNet_v2_utils import SLITNetDataset_white as Dataset
from SLITNet_v2_utils import apply_class_thresholds
from SLITNet_v2_utils import get_predictions_in_small_size as get_predictions
from SLITNet_v2_utils import restrict_within_limbus
from SLITNet_v2_utils import clean_predictions_white as clean_predictions
from SLITNet_v2_utils import calculate_performance_metrics
from SLITNet_v2_utils import initialize_validation_summary_metrics, write_validation_summary_metrics, plot_validation_summary_metrics

# Settings:
MODE = "inference"


def main():

    ########################################### USER INPUT ##########################################

    # System inputs:
    MODEL_DIR = sys.argv[1]
    MODEL_SUBDIR = sys.argv[2]
    MODEL_NUM = int(sys.argv[3])
    K_FOLD = sys.argv[4]
    DATASET_DIR = sys.argv[5]

    ##################################################################################################

    # Thresholds:
    THRESHOLD_CLASS = 0.0
    THRESHOLD_NMS = 0.5
    THRESHOLD_RANGE = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])


    # Rules:
    RULES = [1, 2, 3, 4, 6, 7, 8, 9, 10]

    ################################################################################################

    # Check existence:
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_SUBDIR)):
        print('The folder does not exist: {}'.format(os.path.join(MODEL_DIR, MODEL_SUBDIR)))
        return

    # Config:
    CONFIG_FILENAME = os.path.join(MODEL_DIR, 'config.pickle')
    config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
    if (MODE == "inference"):
        config.set_to_inference_mode(class_threshold=THRESHOLD_CLASS, nms_threshold=THRESHOLD_NMS)
    config.display()

    # Number of classes:
    NUM_CLASSES = config.NUM_CLASSES - 1

    # Number of thresholds:
    NUM_THRESHOLDS = len(THRESHOLD_RANGE)

    # Best thresholds for THRESHOLD_CLASS:
    BEST_THRESHOLD_FILE = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'validation', 'model-{}'.format(MODEL_NUM),
                                             'THRESHOLD_CLASS_best_thresholds.npy')
    if os.path.exists(BEST_THRESHOLD_FILE):
        THRESHOLDS = np.load(BEST_THRESHOLD_FILE)
    else:
        print('THRESHOLD_CLASS_best_thresholds.npy does not exist.')
        return

    ################################################################################################

    # Dataset:
    VAL_FILENAME = os.path.join(DATASET_DIR, 'val_data_{}.mat'.format(K_FOLD))
    val_dataset = Dataset()
    val_dataset.load_dataset(VAL_FILENAME)
    val_dataset.prepare()
    print('Validation dataset: {}'.format(VAL_FILENAME))
    print('Validation dataset prepared: {} images'.format(val_dataset.num_images))

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15 to")
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")

    # Recreate the model in inference mode:
    model = modellib.MaskRCNN(mode=MODE,
                              config=config,
                              model_dir=os.path.join(MODEL_DIR, MODEL_SUBDIR))

    # Check if model exists and load weights:
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'mask_rcnn_ulcer_' + str(MODEL_NUM).zfill(4) + '.h5')
    if os.path.exists(model_path):
        model.load_weights(model_path,
                           by_name=True)
        print('Loading weights from: {}'.format(model_path))
    else:
        print('Model does not exist: {}'.format(model_path))
        return

    # Save folder:
    SAVE_FOLDER = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'validation', 'model-{}'.format(MODEL_NUM))
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    ###############################################################################################################################

    # Keep_track:
    summary_metrics = np.zeros([NUM_CLASSES, NUM_THRESHOLDS])
    summary_count = np.zeros([NUM_CLASSES, 1])

    for image_id in val_dataset.image_ids:

        print('Image {}'.format(image_id))

        # Load image and annotations:
        image_orig, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset=val_dataset,
                                                                                       config=config,
                                                                                       image_id=image_id,
                                                                                       use_mini_mask=False)
        # Load limbus mask and resize:
        limbus_mask = val_dataset.load_limbus_mask(image_id)
        limbus_mask, _, _, _, _ = resize_image(
            limbus_mask[:, :, None],
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        limbus_mask = np.array(limbus_mask[:, :, 0]).astype(np.float32)

        # Run model detection:
        results_from_model = model.detect([image_orig])[0]

        # Get ground truth:
        ground_truth = {"class_ids": gt_class_id, "rois": gt_bbox, "masks": gt_mask}

        # Loop over each class and threshold:
        for class_num in range(1, NUM_CLASSES+1):

            # Check if class is present:
            if np.where(gt_class_id == class_num)[0].shape[0] == 0:
                continue

            # Add to summary count:
            summary_count[class_num-1, 0] += 1

            for threshold_idx, threshold_val in enumerate(THRESHOLD_RANGE):

                # Copy results (so they do not change above) and apply thresholds:
                results = {"class_ids": results_from_model["class_ids"],
                           "rois": results_from_model["rois"],
                           "masks": results_from_model["masks"],
                           "scores": results_from_model["scores"]
                           }
                results = apply_class_thresholds(results, THRESHOLDS)

                # Initialise mask thresholds for each class:
                thresholds_mask = 0.5 * np.ones([NUM_CLASSES, 1])
                thresholds_mask[class_num - 1, 0] = threshold_val

                # Get predictions and clean:
                results = get_predictions(results, thresholds_mask)
                results = restrict_within_limbus(results, IDS_TO_RESTRICT=[1, 2, 4, 6], HIGH_OVERLAP=0.7, LIMBUS_MASK=limbus_mask)
                results = restrict_within_limbus(results, IDS_TO_RESTRICT=[3], HIGH_OVERLAP=0.5, LIMBUS_MASK=limbus_mask)
                results = clean_predictions(results, RULES)

                # Calculate performance metrics:
                class_dsc, _, _ = calculate_performance_metrics(pred=results,
                                                                truth=ground_truth,
                                                                num_classes=NUM_CLASSES)

                # Add to summary metrics:
                summary_metrics[class_num-1, threshold_idx] += class_dsc[class_num-1, 0]

    # Calculate summary metrics:
    summary_metrics = summary_metrics/summary_count

    # Write summary metrics:
    for class_num in range(1, NUM_CLASSES+1):
        SUMMARY_FILENAME = initialize_validation_summary_metrics(SAVE_FOLDER, 'THRESHOLD_MASK', class_num)
        write_validation_summary_metrics(SUMMARY_FILENAME, THRESHOLD_RANGE, summary_metrics, class_num)

    # Plot and save best thresholds:
    best_thresholds = np.zeros(NUM_CLASSES)
    for class_num in range(1, NUM_CLASSES+1):
        best_threshold = plot_validation_summary_metrics(SAVE_FOLDER, 'validation_THRESHOLD_MASK_class-{}'.format(class_num), 'max')
        best_thresholds[class_num-1] = best_threshold
    np.save(os.path.join(SAVE_FOLDER, 'THRESHOLD_MASK_best_thresholds.npy'), best_thresholds)


    print('Finished.')

if __name__ == "__main__":
    main()
