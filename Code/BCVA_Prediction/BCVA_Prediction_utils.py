# SLIT-Net v2
# DOI: 10.1167/tvst.10.12.2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import h5py as hh
import numpy as np


class Dataset(object):

    def __init__(self):

        self.image_info = []
        self.num_images = []
        self.vector_length = None

    def prepare_dataset(self, filename):

        # Get source name:
        source = filename.split('/')
        source = source[len(source) - 1]
        source = source.split('.')
        source = source[0]

        # Open file:
        with hh.File(filename) as f:

            # Get data:
            data = f['data']

            # Get number of measurements:
            nMeasurements = data['MEASUREMENTS'].shape[0]

            # Read data:
            for i in range(nMeasurements):

                # Path:
                white_path = ''
                white_path_ref = data['WHITE_PATH'][i, 0]
                white_path_to_decode = data[white_path_ref].value
                for p in white_path_to_decode:
                    white_path = white_path + chr(p)

                blue_path = ''
                blue_path_ref = data['BLUE_PATH'][i, 0]
                blue_path_to_decode = data[blue_path_ref].value
                for p in blue_path_to_decode:
                    blue_path = blue_path + chr(p)

                # Measurements vector:
                vector_ref = data['MEASUREMENTS'][i, 0]
                vector = data[vector_ref].value
                vector = vector.flatten()

                # Update vector length:
                if self.vector_length is None:
                    self.vector_length = vector.shape[0]
                else:
                    assert vector.shape[0] == self.vector_length, 'All vectors must have the same length'

                # Label:
                label_ref = data['LOGMAR'][i, 0]
                label = data[label_ref].value[0, 0]

                # Add to dataset:
                self.add_image(source=source,
                               image_id=i,
                               white_path=white_path,
                               blue_path=blue_path,
                               vector=vector,
                               label=label)

        # Update dataset:
        self.num_images = len(self.image_info)

    def add_image(self, source, image_id, **kwargs):
        image_info = {
            "id": image_id,
            "source": source
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def get_vectors_and_labels(self):

        # Initialise:
        vectors = np.zeros([self.num_images, self.vector_length])
        labels = np.zeros([self.num_images])

        for image_id in range(self.num_images):
            vectors[image_id, :] = self.image_info[image_id]["vector"]
            labels[image_id] = self.image_info[image_id]["label"]

        return vectors, labels


def get_mean_and_std(filename):
    """
    Mean and standard deviation of measurements vectors
    """

    if not os.path.exists(filename):
        print('{} does not exist.'.format(filename))
        return None, None

    else:
        with hh.File(filename) as f:
            mean = f['mean_vector'].value
            mean = mean.flatten()
            std = f['std_vector'].value
            std = std.flatten()
            return mean, std
