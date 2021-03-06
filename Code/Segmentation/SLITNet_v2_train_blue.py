# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import os
import sys
import time
import pickle
import warnings
import imgaug.augmenters as iaa
import keras

import SLITNet_v2_model as modellib
from SLITNet_v2_utils import SLITNetDataset_blue as Dataset
from SLITNet_v2_utils import SLITNetConfig as Config
from SLITNet_v2_utils import plot_learning_curves

# Settings:
MODE = "training"


def main():

    ########################################### USER INPUT ##########################################

    # Save directory:
    MODEL_DIR = sys.argv[1]
    MODEL_SUBDIR = sys.argv[2]
    WITH_AUGMENTATION = bool(int(sys.argv[3]))
    INIT_WITH = sys.argv[4]
    NUM_EPOCHS = [int(sys.argv[5]), int(sys.argv[6])]
    K_FOLD = sys.argv[7]
    AUGMENT_VAL = bool(int(sys.argv[8]))
    MIN_EPOCH_TO_SAVE = int(sys.argv[9])
    PERIOD_TO_SAVE = int(sys.argv[10])
    DATASET_DIR = sys.argv[11]

    # Note: The sum of NUM_EPOCHS is always the final total number of epochs to train.

    #################################################################################################

    # Config:
    CONFIG_WRITE_FILENAME = os.path.join(MODEL_DIR, 'config.txt')
    CONFIG_SAVE_FILENAME = os.path.join(MODEL_DIR, 'config.pickle')
    if os.path.exists(CONFIG_SAVE_FILENAME):
        config = pickle.load(open(CONFIG_SAVE_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
        print('A config file already exists and will be used. '
              'For new configs, please create a new folder.')
    else:
        config = Config()
        config.write_to_file(filename=CONFIG_WRITE_FILENAME)
        with open(CONFIG_SAVE_FILENAME, 'wb') as cf:
            pickle.dump(config, cf, pickle.HIGHEST_PROTOCOL)
        print('A new config was created.')
    config.display()

    # Check settings:
    if (INIT_WITH == 'scratch' or INIT_WITH == 'last'):
        # Always train 'all' when initialising from 'scratch' or 'last':
        if (NUM_EPOCHS[0] > 0):
            print('When INIT_WITH = {}, heads should not be trained separately.'.format(INIT_WITH))
            return

    # Make directories:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Check what directories already exist:
    if not (INIT_WITH == 'last'):
        existing_folders = next(os.walk(MODEL_DIR))[1]
        if (MODEL_SUBDIR in existing_folders):
            print('The sub-folder {} already exists. '
                  'Please create a new sub-folder.'.format(MODEL_SUBDIR))
            return

    #################################################################################################

    # Write filenames:
    if(INIT_WITH == 'last'):
        CSV_WRITE_FILENAME = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'loss.csv')
    else:
        CSV_WRITE_FILENAME = os.path.join(MODEL_DIR, MODEL_SUBDIR + '.csv')
    SUMMARY_WRITE_FILENAME = os.path.join(MODEL_DIR, 'summary.txt')

    # Datasets:
    dataset_start = time.time()

    TRAIN_FILENAME = os.path.join(DATASET_DIR, 'train_data_{}.mat'.format(K_FOLD))
    train_dataset = Dataset()
    train_dataset.load_dataset(TRAIN_FILENAME)
    train_dataset.prepare()
    print('Training dataset: {}'.format(TRAIN_FILENAME))
    print('Training dataset prepared: {} images'.format(train_dataset.num_images))


    VAL_FILENAME = os.path.join(DATASET_DIR, 'val_data_{}.mat'.format(K_FOLD))
    val_dataset = Dataset()
    val_dataset.load_dataset(VAL_FILENAME)
    val_dataset.prepare()
    print('Validation dataset: {}'.format(VAL_FILENAME))
    print('Validation dataset prepared: {} images'.format(val_dataset.num_images))

    dataset_elapsed = time.time() - dataset_start

    # Augmentation:
    if WITH_AUGMENTATION:
        augmentation = iaa.Sequential([
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
        augmentation = None

    # Keras callbacks:
    custom_callbacks = [keras.callbacks.CSVLogger(CSV_WRITE_FILENAME, separator='\t', append=True)]

    # Create model in training mode:
    model_start = time.time()

    model = modellib.MaskRCNN(mode=MODE,
                              config=config,
                              model_dir=MODEL_DIR)

    # Initialize weights:
    if(INIT_WITH == 'scratch'):
        pass

    elif(INIT_WITH == "last"):
        LAST_CHECKPOINT = model.find_last_within_subdir(os.path.join(MODEL_DIR, MODEL_SUBDIR))
        print('Checkpoint found: {}'.format(LAST_CHECKPOINT))
        model.load_weights(LAST_CHECKPOINT,
                           by_name=True)

    elif(INIT_WITH == "coco"):
        COCO_MODEL_PATH = './mrcnn/mask_rcnn_coco.h5'
        model.load_weights(COCO_MODEL_PATH,
                           by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    else:
        print('Cannot initialize weights with {}'.format(INIT_WITH))
        return
    print('Weights initialized with "{}"'.format(INIT_WITH))

    # Start:
    model_elapsed = time.time() - model_start

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15 to ")

    # Train in two stages:
    #
    #       1. Only the heads. Freeze all the backbone layers and train only the randomly-initialized
    #          layers (i.e. that don't have pre-trained COCO weights).
    #          To do this, pass layers='heads' to train()
    #       2. All layers. Fine-tune all layers.
    #          To do this, pass layers='all' to train()
    #
    train_start = time.time()

    # Total training epochs:
    TOTAL_EPOCHS = NUM_EPOCHS[0] + NUM_EPOCHS[1]

    # Heads:
    if(NUM_EPOCHS[0] > 0):
        model.train(train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=NUM_EPOCHS[0],
                    layers="heads",
                    augmentation=augmentation,
                    custom_callbacks=custom_callbacks,
                    augment_val=AUGMENT_VAL,
                    period_to_save=PERIOD_TO_SAVE)

    # All:
    if(NUM_EPOCHS[1] > 0):
        model.train(train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=TOTAL_EPOCHS,
                    layers="all",
                    augmentation=augmentation,
                    custom_callbacks=custom_callbacks,
                    augment_val=AUGMENT_VAL,
                    period_to_save=PERIOD_TO_SAVE)

    train_elapsed = time.time() - train_start

    # Rename the sub-folder:
    if (INIT_WITH == 'last'):
        temp = MODEL_SUBDIR
    else:
        current_folders = next(os.walk(MODEL_DIR))[1]
        temp = [f for f in current_folders if f not in existing_folders][0]
        os.rename(os.path.join(MODEL_DIR, temp), os.path.join(MODEL_DIR, MODEL_SUBDIR))

    # Write to file:
    with open(SUMMARY_WRITE_FILENAME, 'a') as wf:
        wf.write('Sub-folder: {} | {} | {} | Augmentation: {} \n'.format(MODEL_SUBDIR, temp, INIT_WITH, WITH_AUGMENTATION))
        wf.write('Epochs: {} (heads), {} (all) \n'.format(NUM_EPOCHS[0], NUM_EPOCHS[1]))
        wf.write('Dataset preparation: {} seconds \n'.format(dataset_elapsed))
        wf.write('Model initialization: {} seconds \n'.format(model_elapsed))
        wf.write('Training time for {} epochs: {} seconds \n'.format(TOTAL_EPOCHS, train_elapsed))
        wf.write('Average training time per epoch: {} seconds \n'.format(train_elapsed/TOTAL_EPOCHS))
        wf.write('\n')

    # Move file into sub-folder:
    if not (INIT_WITH == 'last'):
        NEW_CSV_WRITE_FILENAME = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'loss.csv')
        os.rename(CSV_WRITE_FILENAME, NEW_CSV_WRITE_FILENAME)

    # Plot learning curves:
    plot_learning_curves(MODEL_DIR, MODEL_SUBDIR)

    # Delete unnecessary models:
    for m in range(TOTAL_EPOCHS):
        if m < MIN_EPOCH_TO_SAVE:
            filepath = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'mask_rcnn_ulcer_{:0>4d}.h5'.format(m))
            if os.path.exists(filepath):
                os.remove(filepath)

    print('Finished.')

if __name__ == '__main__':
    main()
