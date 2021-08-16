import os
import pickle
from LimbusNet_utils import SegmentationConfig as Config

def main():

    ################################# USER INPUT #################################

    MAIN_DIR = ('/media/ubuntu/My Passport1/Jessica/Michigan_Woodward'
                '/TimeSeries_newPupil_newReflex/For_Github'
                '/Trained_Models/Segmentation/Limbus-Net/Blue_Light')

    ##############################################################################

    # Initialise files:
    CONFIG_WRITE_FILENAME = os.path.join(MAIN_DIR, 'config.txt')
    CONFIG_SAVE_FILENAME = os.path.join(MAIN_DIR, 'config.pickle')

    # Make directory:
    if not os.path.exists(MAIN_DIR):
        os.makedirs(MAIN_DIR)

    # Make configuration:
    if os.path.exists(CONFIG_SAVE_FILENAME):
        print('A config file already exists. '
              'For new configs, please create a new folder.')
        return

    else:
        config = Config()
        with open(CONFIG_SAVE_FILENAME, 'wb') as cf:
            pickle.dump(config, cf, pickle.HIGHEST_PROTOCOL)
        config.display()
        config.write_to_file(filename=CONFIG_WRITE_FILENAME)
        print('New config saved.')

if __name__ == '__main__':
    main()
