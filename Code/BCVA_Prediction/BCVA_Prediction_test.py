import os
import sys
import pickle
import numpy as np
from sklearn.svm import SVR as Model

from BCVA_Prediction_utils import Dataset
from BCVA_Prediction_utils import get_mean_and_std

def main():

    # User-input:
    MAIN_DIR = sys.argv[1]
    DATASET_DIR = sys.argv[2]

    ####################################################################################################################

    # Other settings:
    ROUND_VALUES = True
    CAP_VALUES = True

    ####################################################################################################################

    # Save directory:
    SAVE_DIR = os.path.join(MAIN_DIR, 'Final_Predictions')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Write filename:
    WRITE_FILENAME = os.path.join(SAVE_DIR, 'testing.txt')
    with open(WRITE_FILENAME, 'w') as wf:
        wf.write('K-fold\tIndex\tTruth_Value\tPred_Value')

    for K_TAG in ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']:

        # Dataset:
        TEST_FILENAME = os.path.join(DATASET_DIR, 'test_data_{}.mat'.format(K_TAG))
        test_dataset = Dataset()
        test_dataset.prepare_dataset(TEST_FILENAME)
        test_vectors, test_labels = test_dataset.get_vectors_and_labels()
        NUM_IMAGES = test_dataset.num_images
        print('Testing dataset: {}'.format(TEST_FILENAME))
        print('Testing dataset prepared: {} images'.format(NUM_IMAGES))

        # Normalise:
        mean, std = get_mean_and_std(os.path.join(DATASET_DIR, 'measurements_stats.mat'))
        test_vectors = (test_vectors - mean) / std

        # Model directory:
        MODEL_DIR = os.path.join(MAIN_DIR, K_TAG)

        # Model filename:
        MODEL_FILENAME = os.path.join(MODEL_DIR, 'model-final.pickle')

        # Load model:
        model = pickle.load(open(MODEL_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))

        # Get predictions:
        pred_labels = model.predict(test_vectors)

        for image_id in range(NUM_IMAGES):

            # Get values:
            pred_value = pred_labels[image_id]
            truth_value = test_labels[image_id]

            if CAP_VALUES:
                if pred_value < 0.0:
                    pred_value = 0.0
                elif pred_value > 2.3:
                    pred_value = 2.3

            if ROUND_VALUES:
                pred_value = np.round(pred_value, 1)
                truth_value = np.round(truth_value, 1)

            # Write to file:
            with open(WRITE_FILENAME, 'a') as wf:
                wf.write('\n{}\t{}\t{}\t{}'.format(K_TAG, image_id, truth_value, pred_value))


    print('Finished.')

if __name__ == '__main__':
    main()