import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

import os
import sys
import tensorflow as tf
import numpy as np
import warnings
import pickle
import skimage.transform
import skimage.measure
import scipy.io as sio

import LimbusNet_utils as utils
import LimbusNet_model as model

######################################### SETTINGS ####################################################

# Mode:
IS_TRAINING = False

# Classes:
NUM_CLASSES = 2

# Limbus:
LIMBUS_ID = 4


########################################## MAIN #######################################################

def main():

    # System inputs:
    MAIN_DIR = sys.argv[1]
    K_TAG = sys.argv[2]
    MODEL_NUM = int(sys.argv[3])
    THRESHOLD = float(sys.argv[4])
    DATASET_DIR = sys.argv[5]

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
        IMAGE_MEAN = np.array([19.0, 50.9, 67.8])

    # Load dataset:
    TEST_FILENAME = os.path.join(DATASET_DIR, 'test_data_{}.mat'.format(K_TAG))
    all_images, all_masks = utils.read_mat_dataset(TEST_FILENAME, LIMBUS_ID, include_without_limbus=True)
    TEST_IMAGES = len(all_images)
    print('\nNumber of testing images: {}'.format(TEST_IMAGES))

    # Model directory:
    MODEL_DIR = os.path.join(MAIN_DIR, K_TAG)

    # Save directory:
    SAVE_DIR = os.path.join(MODEL_DIR, 'testing', 'model-{}'.format(MODEL_NUM))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Placeholders:
    mode = tf.placeholder(tf.bool)
    image_batch = tf.placeholder(tf.float32, [1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3])

    # Network:
    logits, _, _, _ = model.segment(image_batch, mode, config.NUM_FEATURES, config.POOL_SIZE, config.REGULARIZATION_WEIGHT)

    # Saver:
    saver = tf.train.Saver(max_to_keep=0)

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
            print('\nmodel-{} does not exist.')
            return

        for im in range(TEST_IMAGES):

            print('\nImage {}'.format(im))

            # Get image and mask:
            original_image_val = all_images[im]
            original_mask_val = all_masks[im]

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
            logits_val = logits_val[0, :, :, :]
            original_mask_val = original_mask_val[:, :, 0]

            # Convert logits to probability:
            prob_val = utils.softmax(logits_val)[:, :, 1]

            # Resize probability:
            original_prob_val = skimage.transform.resize(prob_val,
                                                         (original_mask_val.shape[0], original_mask_val.shape[1]),
                                                         order=1, mode="constant", preserve_range=True)

            # Convert probability to binary:
            original_pred_val = (original_prob_val > THRESHOLD)

            # Postprocess:
            original_pred_val = utils.postprocess(original_pred_val)

            # Save predicted mask:
            sio.savemat(os.path.join(SAVE_DIR, str(im).zfill(3) + '.mat'), {'predicted_mask': original_pred_val})


    print('\nFinished.')


if __name__ == '__main__':
    main()