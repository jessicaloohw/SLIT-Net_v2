# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys
import tensorflow as tf
import numpy as np
import warnings
import pickle
import imgaug.augmenters as iaa
import sklearn.utils

import LimbusNet_utils as utils
import LimbusNet_model as model

######################################### SETTINGS ####################################################

# Mode:
IS_TRAINING = True

# Classes:
NUM_CLASSES = 2

# Limbus:
LIMBUS_ID = 4

########################################## MAIN #######################################################

def main():

    # System inputs:
    MAIN_DIR = sys.argv[1]
    K_TAG = sys.argv[2]
    NUM_EPOCHS = int(sys.argv[3])
    EPOCH_STEPS_TO_SAVE = int(sys.argv[4])
    MIN_EPOCH_TO_SAVE = int(sys.argv[5])
    CONTINUE_TRAINING = bool(int(sys.argv[6]))
    LAST_EPOCH = int(sys.argv[7])
    DATASET_DIR = sys.argv[8]

    ###################################################################################################

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default")

    # Main directory:
    if not os.path.exists(MAIN_DIR):
        os.makedirs(MAIN_DIR)

    # Config:
    CONFIG_FILENAME = os.path.join(MAIN_DIR, 'config.pickle')
    CONFIG_WRITE_FILENAME = os.path.join(MAIN_DIR, 'config.txt')

    if os.path.exists(CONFIG_FILENAME):
        config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
        print('\nA config file exists and will be used.')
        config.display()
        if not os.path.exists(CONFIG_WRITE_FILENAME):
            config.write_to_file(CONFIG_WRITE_FILENAME)
    else:
        config = utils.SegmentationConfig()
        with open(CONFIG_FILENAME, 'wb') as cf:
            pickle.dump(config, cf, pickle.HIGHEST_PROTOCOL)
        print('\nA new config was created with the current settings.')
        config.display()
        config.write_to_file(CONFIG_WRITE_FILENAME)

    # Check settings:
    if not CONTINUE_TRAINING:
        LAST_EPOCH = 0

    if config.SUB_BATCHES > 1:
        print('ERROR: config.SUB_BATCHES > 1 not supported.')
        return

    # Augmentation:
    if config.WITH_AUGMENTATION:
        AUGMENTATION_TYPES = iaa.Sequential([
            iaa.OneOf([iaa.Noop(),
                       iaa.Fliplr(1.0),
                       iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="constant", pad_cval=0),
                       iaa.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-5, 5), shear=0,
                                  mode="constant", cval=0)]),
            iaa.OneOf([iaa.Noop(),
                       iaa.Add((-25, 25), per_channel=0.0),
                       iaa.AdditiveGaussianNoise(scale=(0, 5), per_channel=0.0),
                       iaa.Multiply((0.75, 1.5), per_channel=0.0),
                       iaa.ContrastNormalization((0.75, 1.25), per_channel=0.0)])
        ])
    else:
        AUGMENTATION_TYPES = None

    # Mean subtraction:
    if config.SUBTRACT_MEAN:
        IMAGE_MEAN = np.array([19.0, 50.9, 67.8])

    # Load dataset:
    TRAIN_FILENAME = os.path.join(DATASET_DIR, 'train_data_{}.mat'.format(K_TAG))
    all_images, all_masks = utils.read_mat_dataset(TRAIN_FILENAME, LIMBUS_ID)
    TRAIN_IMAGES = len(all_images)
    print('\nNumber of training images: {}'.format(TRAIN_IMAGES))

    # Save directory:
    SAVE_DIR = os.path.join(MAIN_DIR, K_TAG)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        if not CONTINUE_TRAINING:
            print('{} exists. Please check!'.format(SAVE_DIR))
            return

    # Calculate dimensions:
    IMAGE_HEIGHT1 = config.IMAGE_HEIGHT
    IMAGE_HEIGHT2 = int(IMAGE_HEIGHT1 / config.POOL_SIZE[0])
    IMAGE_HEIGHT3 = int(IMAGE_HEIGHT2 / config.POOL_SIZE[1])
    IMAGE_HEIGHT4 = int(IMAGE_HEIGHT3 / config.POOL_SIZE[2])

    IMAGE_WIDTH1 = config.IMAGE_WIDTH
    IMAGE_WIDTH2 = int(IMAGE_WIDTH1 / config.POOL_SIZE[0])
    IMAGE_WIDTH3 = int(IMAGE_WIDTH2 / config.POOL_SIZE[1])
    IMAGE_WIDTH4 = int(IMAGE_WIDTH3 / config.POOL_SIZE[2])

    # Placeholders:
    mode = tf.placeholder(tf.bool)
    image_batch = tf.placeholder(tf.float32, [config.BATCH_SIZE, IMAGE_HEIGHT1, IMAGE_WIDTH1, 3])
    mask_batch1 = tf.placeholder(tf.float32, [config.BATCH_SIZE, IMAGE_HEIGHT1, IMAGE_WIDTH1, 1])
    mask_batch2 = tf.placeholder(tf.float32, [config.BATCH_SIZE, IMAGE_HEIGHT2, IMAGE_WIDTH2, 1])
    mask_batch3 = tf.placeholder(tf.float32, [config.BATCH_SIZE, IMAGE_HEIGHT3, IMAGE_WIDTH3, 1])
    mask_batch4 = tf.placeholder(tf.float32, [config.BATCH_SIZE, IMAGE_HEIGHT4, IMAGE_WIDTH4, 1])

    # Model:
    logits1, logits2, logits3, logits4 = model.segment(image_batch, mode, config.NUM_FEATURES, config.POOL_SIZE, config.REGULARIZATION_WEIGHT)
    hausdorff, dice, hausdorff_dice, reg, total = model.loss(logits1, logits2, logits3, logits4, mask_batch1, mask_batch2, mask_batch3, mask_batch4, config.LOSS_PARAMS)

    # Update batch normalization variables:
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Optimizer:
    opt = tf.train.MomentumOptimizer(learning_rate=config.LEARNING_RATE,
                                     momentum=config.MOMENTUM,
                                     use_nesterov=config.USE_NESTEROV)
    train_op = opt.minimize(total)

    # Saver:
    saver = tf.train.Saver(max_to_keep=0)

    # Summary:
    SUMMARY_DIR = os.path.join(SAVE_DIR, 'training', 'epochs_{}-{}'.format(LAST_EPOCH+1, NUM_EPOCHS))
    if not (os.path.exists(SUMMARY_DIR)):
        os.makedirs(SUMMARY_DIR)

    # Write filenames:
    WRITE_FILENAME = os.path.join(SUMMARY_DIR, 'training.txt')
    if not (CONTINUE_TRAINING):
        with open(WRITE_FILENAME, 'a') as wf:
            wf.write('Epoch\tHausdorff\tDice\tHausdorff_Dice\tRegularisation\tTotal')

    # Session configuration:
    sessConfig = tf.ConfigProto()
    sessConfig.allow_soft_placement = True
    sessConfig.gpu_options.allow_growth = True

    # Session:
    with tf.Session(config=sessConfig) as sess:

        # Model:
        if (CONTINUE_TRAINING):
            PATH_TO_MODEL = SAVE_DIR + '/model-' + str(LAST_EPOCH)
            saver.restore(sess, PATH_TO_MODEL)
            print('\nModel restored: {}'.format(PATH_TO_MODEL))
        else:
            sess.run(tf.global_variables_initializer())
            print('\nModel initialised.')

        # Steps required:
        TRAIN_STEPS = np.floor(float(TRAIN_IMAGES) / (config.SUB_BATCHES * config.BATCH_SIZE)).astype(np.int32)
        print('\nTraining steps per epoch: {}\n'.format(TRAIN_STEPS))

        # Training:
        epochs = []
        hausdorff_loss = []
        hausdorff_dice_loss = []
        dice_loss = []
        reg_loss = []
        total_loss = []

        for epoch in range(LAST_EPOCH + 1, NUM_EPOCHS + 1):

            # Shuffle dataset:
            all_images, all_masks = sklearn.utils.shuffle(all_images, all_masks)
            batch_num = -1

            # Training steps:
            for train_step in range(TRAIN_STEPS):

                accum_hausdorff = 0
                accum_dice = 0
                accum_hausdorff_dice = 0
                accum_reg = 0
                accum_total = 0

                for train_batch in range(config.SUB_BATCHES):

                    # Get image_batch and mask_batch:
                    batch_num += 1
                    sidx = batch_num*config.BATCH_SIZE
                    eidx = (batch_num+1)*config.BATCH_SIZE
                    current_image_batch, current_mask_batch1, current_mask_batch2, current_mask_batch3, current_mask_batch4 = utils.get_current_batch_multiscale(
                        all_images[sidx:eidx], all_masks[sidx:eidx],
                        [IMAGE_HEIGHT1, IMAGE_HEIGHT2, IMAGE_HEIGHT3, IMAGE_HEIGHT4],
                        [IMAGE_WIDTH1, IMAGE_WIDTH2, IMAGE_WIDTH3, IMAGE_WIDTH4],
                        AUGMENTATION_TYPES)

                    if config.SUBTRACT_MEAN:
                        current_image_batch = current_image_batch - IMAGE_MEAN

                    # Training step:
                    _, _, hausdorff_val, dice_val, hausdorff_dice_val, reg_val, total_val = sess.run(
                        [update_op, train_op, hausdorff, dice, hausdorff_dice, reg, total],
                        feed_dict={mode: IS_TRAINING,
                                   image_batch: current_image_batch,
                                   mask_batch1: current_mask_batch1,
                                   mask_batch2: current_mask_batch2,
                                   mask_batch3: current_mask_batch3,
                                   mask_batch4: current_mask_batch4})

                    # Keep track of losses:
                    accum_hausdorff += hausdorff_val
                    accum_dice += dice_val
                    accum_hausdorff_dice += hausdorff_dice_val
                    accum_reg += reg_val
                    accum_total += total_val

                # Calculate average for step:
                average_hausdorff = accum_hausdorff / config.SUB_BATCHES
                average_dice = accum_dice / config.SUB_BATCHES
                average_hausdorff_dice = accum_hausdorff_dice / config.SUB_BATCHES
                average_reg = accum_reg / config.SUB_BATCHES
                average_total = accum_total / config.SUB_BATCHES

                # Print to console to see progress:
                print('Epoch %d/%d \t Step %d/%d \t| Hausdorff: %g \t Dice: %g \t Hausdorff_Dice: %g \t Reg: %g \t Total: %g' % (
                    epoch, NUM_EPOCHS, train_step + 1, TRAIN_STEPS, average_hausdorff, average_dice, average_hausdorff_dice, average_reg, average_total))

            # Keep track after each epoch:
            epochs.append(epoch)
            hausdorff_loss.append(average_hausdorff)
            dice_loss.append(average_dice)
            hausdorff_dice_loss.append(average_hausdorff_dice)
            reg_loss.append(average_reg)
            total_loss.append(average_total)

            # Write to file after each epoch:
            write_string = ('%d\t%g\t%g\t%g\t%g\t%g' % (
                epoch, average_hausdorff, average_dice, average_hausdorff_dice, average_reg, average_total))
            with open(WRITE_FILENAME, 'a') as wf:
                wf.write('\n' + write_string)

            # Save after each epoch (if required):
            if ((epoch >= MIN_EPOCH_TO_SAVE) and (epoch % EPOCH_STEPS_TO_SAVE == 0)) or (epoch == NUM_EPOCHS):
                save_path = saver.save(sess, SAVE_DIR + '/model', epoch)
                print('model-%d saved: %s' % (epoch, save_path))

            # Plot learning curves:
            if (epoch % 10 == 0) or (epoch == NUM_EPOCHS):
                utils.plot_learning_curves(SUMMARY_DIR, epochs, hausdorff_loss, None, 'hausdorff')
                utils.plot_learning_curves(SUMMARY_DIR, epochs, dice_loss, None, 'dice')
                utils.plot_learning_curves(SUMMARY_DIR, epochs, hausdorff_dice_loss, None, 'hausdorff_dice')
                utils.plot_learning_curves(SUMMARY_DIR, epochs, reg_loss, None, 'regularisation')
                utils.plot_learning_curves(SUMMARY_DIR, epochs, total_loss, None, 'total')

        print('\nFinished.')


if __name__ == '__main__':
    main()
