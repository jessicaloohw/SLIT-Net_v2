import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys
import tensorflow as tf
import numpy as np
import warnings
import pickle
import skimage.transform
import skimage.measure

import LimbusNet_utils as utils
import LimbusNet_model as model

######################################### SETTINGS ####################################################

# Mode:
IS_TRAINING = False

# Classes:
NUM_CLASSES = 2

# Threshold:
THRESHOLD_RANGE = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                            0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                            0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])

# Number of thresholds:
NUM_THRESHOLDS = len(THRESHOLD_RANGE)

# Limbus:
LIMBUS_ID = 7

########################################## MAIN #######################################################

def main():

    # System inputs:
    MAIN_DIR = sys.argv[1]
    K_TAG = sys.argv[2]
    MODEL_NUM = int(sys.argv[3])
    DATASET_DIR = sys.argv[4]

    ####################################################################################################

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default")

    # Config:
    CONFIG_FILENAME = os.path.join(MAIN_DIR, 'config.pickle')
    CONFIG_WRITE_FILENAME = os.path.join(MAIN_DIR, 'config.txt')

    if os.path.exists(CONFIG_FILENAME):
        config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
        print('\nA config file exists and will be used.')
        if not os.path.exists(CONFIG_WRITE_FILENAME):
            config.write_to_file(CONFIG_WRITE_FILENAME)
    else:
        print('\nNo config file exists. Please check.')
        return

    # Mean subtraction:
    if config.SUBTRACT_MEAN:
        IMAGE_MEAN = np.array([103.0, 68.8, 50.3])

    # Load dataset:
    VAL_FILENAME = os.path.join(DATASET_DIR, 'val_data_{}.mat'.format(K_TAG))
    all_images, all_masks = utils.read_mat_dataset(VAL_FILENAME, LIMBUS_ID)
    VAL_IMAGES = len(all_images)
    print('\nNumber of validation images: {}'.format(VAL_IMAGES))

    # Model directory:
    MODEL_DIR =  os.path.join(MAIN_DIR, K_TAG)

    # Save directory:
    SAVE_DIR = os.path.join(MODEL_DIR, 'validation', 'model-{}'.format(MODEL_NUM))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Placeholders:
    mode = tf.placeholder(tf.bool)
    image_batch = tf.placeholder(tf.float32, [1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])

    # Model:
    logits, _, _, _ = model.segment(image_batch, mode, config.NUM_FEATURES, config.POOL_SIZE, config.REGULARIZATION_WEIGHT)

    # Saver:
    saver = tf.train.Saver(max_to_keep=0)

    # Write filename:
    WRITE_FILENAME = os.path.join(SAVE_DIR, 'validation.txt')
    if not os.path.exists(WRITE_FILENAME):
        with open(WRITE_FILENAME, 'a') as wf:
            wf.write('Threshold\tDSC_Metric')

    # Session configuration:
    sessConfig = tf.ConfigProto()
    sessConfig.allow_soft_placement = True
    sessConfig.gpu_options.allow_growth = True

    # Session:
    with tf.Session(config=sessConfig) as sess:

        # Load model:
        PATH_TO_MODEL = MODEL_DIR + '/model-{}'.format(MODEL_NUM)
        if os.path.exists('{}.meta'.format(PATH_TO_MODEL)):
            saver.restore(sess, PATH_TO_MODEL)
            print('\nmodel-{} restored'.format(MODEL_NUM))
        else:
            print('\nmodel-{} does not exist.'.format(MODEL_NUM))
            return

        # Keep track:
        accum_dsc = np.zeros([NUM_THRESHOLDS])

        for val_step in range(VAL_IMAGES):

            print('\nImage {}'.format(val_step))

            # Get image and mask:
            original_image_val = all_images[val_step]
            original_mask_val = all_masks[val_step]

            # Resize image:
            image_val = skimage.transform.resize(original_image_val.astype(np.uint8),
                                                 (config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
                                                 order=1, mode="constant", preserve_range=True)
            image_val = image_val.astype(np.float32)
            image_val = image_val[None, :, :, :]

            if config.SUBTRACT_MEAN:
                image_val = image_val - IMAGE_MEAN

            # Get prediction:
            logits_val = sess.run(logits,
                                  feed_dict={mode: IS_TRAINING,
                                             image_batch: image_val})

            # Remove dimensions:
            logits_val = logits_val[0,:,:,:]
            original_mask_val = original_mask_val[:,:, 0]

            # Convert logits to probability:
            prob_val = utils.softmax(logits_val)[:, :, 1]

            # Resize probability:
            original_prob_val = skimage.transform.resize(prob_val,
                                                         (original_mask_val.shape[0], original_mask_val.shape[1]),
                                                         order=1, mode="constant", preserve_range=True)

            # Loop through thresholds:
            for threshold_idx, threshold_val in enumerate(THRESHOLD_RANGE):

                # Convert probability to binary:
                original_pred_val = (original_prob_val > threshold_val)

                # Postprocess:
                original_pred_val = utils.postprocess(original_pred_val)

                # Calculate DSC:
                dsc_val = utils.calculate_dsc(original_mask_val, original_pred_val)

                # Keep track:
                accum_dsc[threshold_idx] += dsc_val

        # Calculate average:
        average_dsc = accum_dsc / VAL_IMAGES

        # Write to file:
        for threshold_idx, threshold_val in enumerate(THRESHOLD_RANGE):
            with open(WRITE_FILENAME, 'a') as wf:
                wf.write('\n{}\t{}'.format(threshold_val, average_dsc[threshold_idx]))

        # Plot curves and get best threshold:
        utils.plot_validation_summary_metrics(SAVE_DIR, THRESHOLD_RANGE, average_dsc, 'max', 'max', 'average_dsc_metric')

        print('\nFinished.')


if __name__ == '__main__':
    main()