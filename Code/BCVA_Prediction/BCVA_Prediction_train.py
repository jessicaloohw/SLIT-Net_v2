import os
import sys
import pickle
from sklearn.svm import SVR as Model

from BCVA_Prediction_utils import Dataset
from BCVA_Prediction_utils import get_mean_and_std

def main():

    # User-input:
    MAIN_DIR = sys.argv[1]
    DATASET_DIR = sys.argv[2]

    ####################################################################################################################

    # SVM model parameters:
    parameters = {'kernel': 'rbf',
                  'gamma': 'scale',
                  'tol': 0.001,
                  'C': 0.5,             # Inverse of regularization term
                  'epsilon': 0.05,
                  'verbose': False,
                  'max_iter': 3000
                  }

    ####################################################################################################################

    for K_TAG in ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']:

        # Dataset:
        TRAIN_FILENAME = os.path.join(DATASET_DIR, 'train_data_{}.mat'.format(K_TAG))
        train_dataset = Dataset()
        train_dataset.prepare_dataset(TRAIN_FILENAME)
        train_vectors, train_labels = train_dataset.get_vectors_and_labels()
        print('Training dataset: {}'.format(TRAIN_FILENAME))
        print('Training dataset prepared: {} images'.format(train_dataset.num_images))

        # Normalize:
        mean, std = get_mean_and_std(os.path.join(DATASET_DIR,'measurements_stats.mat'))
        train_vectors = (train_vectors - mean) / std

        # Model directory:
        MODEL_DIR = os.path.join(MAIN_DIR, K_TAG)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # Create model:
        model = Model()
        model.set_params(**parameters)

        # Train model:
        model.fit(train_vectors, train_labels)

        # Save model:
        SAVE_FILENAME = os.path.join(MODEL_DIR, 'model-final.pickle')
        with open(SAVE_FILENAME, 'wb') as sf:
            pickle.dump(model, sf, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()