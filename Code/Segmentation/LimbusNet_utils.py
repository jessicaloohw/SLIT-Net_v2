# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py as hh
import imgaug
import skimage.io
import skimage.transform
import skimage.segmentation
import skimage.measure
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial.distance import directed_hausdorff
from matplotlib import patches
from matplotlib.patches import Polygon

import tensorflow as tf

################################################# CONFIG ###############################################

class SegmentationConfig(object):

    def __init__(self):

        ############################ IMAGE ######################################

        # Training parameters:
        self.IMAGE_HEIGHT = 1024
        self.IMAGE_WIDTH = 1536
        self.SUBTRACT_MEAN = True
        self.WITH_AUGMENTATION = True

        ###################### SEGMENTATION NETWORK ##############################

        # Architecture parameters:
        self.NUM_FEATURES = 16
        self.POOL_SIZE = [4, 4, 4, 4]

        # Training parameters:
        self.SUB_BATCHES = 1
        self.BATCH_SIZE = 3
        self.LOSS_PARAMS = {'hausdorff_power': 2.0}
        self.MOMENTUM = 0.9
        self.USE_NESTEROV = True
        self.LEARNING_RATE = 0.001
        self.REGULARIZATION_WEIGHT = 0.001

        ########################### OTHERS ######################################

        # Comments:
        self.COMMENTS = 'Final settings.'

        # Derived parameters:
        self.QUEUE_PARAMS = {'batch_size': self.BATCH_SIZE,
                             'num_threads': 1,
                             'capacity': 2 * self.SUB_BATCHES * self.BATCH_SIZE,
                             'min_after_dequeue': self.BATCH_SIZE}

    def set_to_test_mode(self):
        self.QUEUE_PARAMS['batch_size'] = 1

    def display(self):
        print('\n\nConfigurations:')
        for a in dir(self):
            if not a.startswith('__') and not callable(getattr(self, a)):
                print('{:30} {}'.format(a, getattr(self, a)))
        print('\n\n')

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('Configurations:\n')
            for a in dir(self):
                if not a.startswith('__') and not callable(getattr(self, a)):
                    f.write('{:30} {} \n'.format(a, getattr(self, a)))


####################################### DATA LOADING ##################################################

def read_mat_dataset(path_to_mat, limbus_id, id=None, include_without_limbus=False):
    """
    If no ID specified, read everything into lists (because some images will have different dimensions)
    """

    # Read data from file:
    data = hh.File(path_to_mat)['data']

    if id is not None:

        print('Loading ID {} from: {}'.format(id, path_to_mat))

        # Initialise:
        mask = None

        # Path to image:
        path = ''
        path_ref = data['PATH'][id, 0]
        path_to_decode = data[path_ref].value
        for p in path_to_decode:
            path = path + chr(p)

        # Read image:
        image = skimage.io.imread(path)

        # References:
        labels_ref = data['LABEL'][id, 0]
        masks_ref = data['MASK'][id, 0]

        # For each object:
        nObjects = data[labels_ref].shape[0]

        for j in range(nObjects):

            # Check label:
            label_ref = data[labels_ref].value[j, 0]
            label = data[label_ref].value

            # Add if limbus:
            if label == limbus_id:
                mask_ref = data[masks_ref].value[j, 0]
                mask = data[mask_ref].value
                mask = np.transpose(mask, (1, 0))

        # Add dimensions:
        if mask is not None:
            mask = mask[:, :, None]
            return image.astype(np.float32), mask.astype(np.float32)

        else:
            return image.astype(np.float32), None

    else:

        print('Loading dataset: {}'.format(path_to_mat))

        # Get number of images:
        nImages = data['IMAGE'].shape[0]

        # Initialise:
        images = []
        masks = []

        # Read data:
        for i in range(nImages):

            # Limbus found:
            limbus_found = False

            # Path to image:
            path = ''
            path_ref = data['PATH'][i, 0]
            path_to_decode = data[path_ref].value
            for p in path_to_decode:
                path = path + chr(p)

            # Read image:
            image = skimage.io.imread(path)

            # References:
            labels_ref = data['LABEL'][i, 0]
            masks_ref = data['MASK'][i, 0]

            # For each object:
            nObjects = data[labels_ref].shape[0]

            for j in range(nObjects):

                # Check label:
                label_ref = data[labels_ref].value[j, 0]
                label = data[label_ref].value

                # Add if limbus:
                if label == limbus_id:
                    mask_ref = data[masks_ref].value[j, 0]
                    mask = data[mask_ref].value
                    mask = np.transpose(mask, (1, 0))
                    mask = mask[:, :, None]

                    images.append(image.astype(np.float32))
                    masks.append(mask.astype(np.float32))
                    limbus_found = True

            # If none of the objects are limbus but image is still required:
            if include_without_limbus:
                if not limbus_found:
                    mask = np.zeros([image.shape[0], image.shape[1]])
                    mask = mask[:, :, None]

                    images.append(image.astype(np.float32))
                    masks.append(mask.astype(np.float32))

        return images, masks


def get_current_batch_multiscale(images, masks, resize_height, resize_width, augmentation):
    """
    Resizes and applies augmentation to images and masks of current batch
    """

    # Make sure same length:
    assert len(resize_height) == 4
    assert len(resize_width) == 4
    assert len(images) == len(masks), "There must be the same number of images and masks in each batch!"
    batch_size = len(images)

    # Initialise:
    batch_images = np.zeros([batch_size, resize_height[0], resize_width[0], 3])
    batch_masks1 = np.zeros([batch_size, resize_height[0], resize_width[0], 1])
    batch_masks2 = np.zeros([batch_size, resize_height[1], resize_width[1], 1])
    batch_masks3 = np.zeros([batch_size, resize_height[2], resize_width[2], 1])
    batch_masks4 = np.zeros([batch_size, resize_height[3], resize_width[3], 1])

    for i in range(batch_size):

        # Get image and mask:
        image = images[i]
        mask = masks[i]

        # Resize:
        image = skimage.transform.resize(image.astype(np.uint8),
                                         (resize_height[0], resize_width[0]),
                                         order=1, mode="constant", preserve_range=True)
        mask = skimage.transform.resize(mask.astype(np.bool),
                                        (resize_height[0], resize_width[0]),
                                        order=1, mode="constant", preserve_range=True)

        # Make sure of units:
        image = image.astype(np.uint8)
        mask = mask.astype(np.bool)

        # Augment (if required):
        if augmentation is not None:

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8),
                                     hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Convert to bool:
            mask = mask.astype(np.bool)

        # Resize to other sizes:
        mask2 = skimage.transform.resize(mask.astype(np.bool),
                                         (resize_height[1], resize_width[1]),
                                         order=1, mode="constant", preserve_range=True)
        mask3 = skimage.transform.resize(mask.astype(np.bool),
                                         (resize_height[2], resize_width[2]),
                                         order=1, mode="constant", preserve_range=True)
        mask4 = skimage.transform.resize(mask.astype(np.bool),
                                         (resize_height[3], resize_width[3]),
                                         order=1, mode="constant", preserve_range=True)

        # Add to batch:
        batch_images[i] = image
        batch_masks1[i] = mask.astype(np.bool)
        batch_masks2[i] = mask2.astype(np.bool)
        batch_masks3[i] = mask3.astype(np.bool)
        batch_masks4[i] = mask4.astype(np.bool)

    # Convert to np.array and float32:
    batch_images = np.array(batch_images).astype(np.float32)
    batch_masks1 = np.array(batch_masks1).astype(np.float32)
    batch_masks2 = np.array(batch_masks2).astype(np.float32)
    batch_masks3 = np.array(batch_masks3).astype(np.float32)
    batch_masks4 = np.array(batch_masks4).astype(np.float32)
    return batch_images, batch_masks1, batch_masks2, batch_masks3, batch_masks4

####################################### POSTPROCESSING ################################################

def postprocess(mask):
    """
    Keeps largest connected component and fills holes
    """

    # Initialise:
    cleaned_mask = np.zeros_like(mask)
    max_area = 0
    max_idx = -1

    # Fill holes:
    mask = binary_fill_holes(mask)

    # Keep largest component only:
    labeled = skimage.measure.label(mask)
    cc = skimage.measure.regionprops(labeled)
    for n, region in enumerate(cc):
        if (region.area > max_area):
            max_area = region.area
            max_idx = n

    # Update mask:
    if(max_idx > -1):
        cleaned_mask = (labeled == cc[max_idx].label)
    return cleaned_mask

########################################## METRICS ####################################################

def calculate_dsc(truth_mask, pred_mask):
    """
    :param truth_mask: [IMAGE_HEIGHT, IMAGE_WIDTH]
    :param pred_mask: [IMAGE_HEIGHT, IMAGE_WIDTH]
    """
    intersect_mask = np.multiply(pred_mask, truth_mask)
    pred_area = np.sum(pred_mask)
    truth_area = np.sum(truth_mask)
    intersection = np.sum(intersect_mask)
    union = pred_area + truth_area

    if (union == 0):
        dice = 1.0
    else:
        dice = (2 * intersection) / union
    return dice

####################################### HELPER FUNCTIONS ##############################################

def plot_learning_curves(save_dir, epochs, train_loss, val_loss, title=''):

    epochs = np.array(epochs)
    if not train_loss is None:
        train_loss = np.array(train_loss)
    if not val_loss is None:
        val_loss = np.array(val_loss)

    plt.figure()
    if not train_loss is None:
        plt.plot(epochs, train_loss, 'b', label='training')
    if not val_loss is None:
        plt.plot(epochs, val_loss, 'r', label='validation')
    plt.legend()
    plt.title('{}'.format(title))
    plt.savefig(os.path.join(save_dir, '{}.png'.format(title)))
    plt.close()


def plot_validation_summary_metrics(save_dir, thresholds, metric_values, best_metric_type, best_threshold_type, title):

    # Remove NaNs:
    keep_idxs = ~np.isnan(metric_values)
    thresholds = thresholds[keep_idxs]
    metric_values = metric_values[keep_idxs]

    # Find best threshold:
    if best_threshold_type == 'min':

        if best_metric_type == 'min':
            best_threshold = thresholds[np.argmin(metric_values)]

        elif best_metric_type == 'max':
            best_threshold = thresholds[np.argmax(metric_values)]

    elif best_threshold_type == 'max':

        if best_metric_type == 'min':
            best_threshold = thresholds[len(metric_values) - np.argmin(np.flip(metric_values)) - 1]

        elif best_metric_type == 'max':
            best_threshold = thresholds[len(metric_values) - np.argmax(np.flip(metric_values)) - 1]

    plt.figure()
    plt.plot(thresholds, metric_values)
    plt.title('best ({}) @ THRESHOLD = {:.2f}'.format(best_metric_type, best_threshold))
    plt.savefig(os.path.join(save_dir, '{}.png'.format(title)))
    plt.close()


def softmax(x):
    x = np.exp(x)
    sum_x = np.sum(x, axis=-1)
    sum_x = sum_x[..., None]
    return x/sum_x
