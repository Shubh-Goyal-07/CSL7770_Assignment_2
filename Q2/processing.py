import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
import argparse
import json
import sys
import logging

DATASET_PATH = "data"
SPECTROGRAM_PATH = "spectrograms"
MFCC_PATH = "mfccs"
FEATURES_PATH = "features"
LANG_INFO_PLOT_PATH = "lang_info_plots"
NMFCC = 13


def extract_mfcc(file_path, lang):

    y, sr = librosa.load(f"{DATASET_PATH}/{lang}/{file_path}")

    # Extracting MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=NMFCC)

    mfcc_stats = {}

    for i in range(NMFCC):
        mfcc_stats[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        mfcc_stats[f'mfcc_{i}_std'] = np.std(mfccs[i])
        mfcc_stats[f'mfcc_{i}_min'] = np.min(mfccs[i])
        mfcc_stats[f'mfcc_{i}_max'] = np.max(mfccs[i])
        # mfcc_stats[f'mfcc_{i}_median'] = np.median(mfccs[i])

    return mfccs, mfcc_stats


def make_save_spectrogram(mfccs, lang, file_path):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', cmap='magma')
    plt.savefig(f'{SPECTROGRAM_PATH}/{lang}/{file_path[:-4]}.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def save_audio_mfcc(mfccs, lang, file_path):
    np.save(f'{MFCC_PATH}/{lang}/{file_path[:-4]}.npy', mfccs)


def save_all_lang_plots(all_lang_stats):
    languages = list(all_lang_stats.keys())

    for i in range(NMFCC):
        plt.figure(figsize=(16, 8))
        
        for metric in ['mean', 'std', 'min', 'max']:
            values = [all_lang_stats[lang][f"mfcc_{i}_{metric}"] for lang in languages]
            plt.plot(languages, values, marker='o', linestyle='-', label=f'{metric.capitalize()}')
        
        plt.xlabel('Language')
        plt.ylabel('Value')
        plt.title(f'MFCC {i} Metrics across Languages')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(LANG_INFO_PLOT_PATH, f'mfcc_{i}_comparison.png'))
        plt.close()



def save_all_lang_stats(all_mfcc_stats):
    lang_stats = {}

    for lang in all_mfcc_stats.keys():
        df = pd.DataFrame(all_mfcc_stats[lang])

        # remove file column
        df = df.drop(columns=['file'])

        # take mean of all the stats
        df = df.mean()

        # add stats to lang_stats as a dictionary
        lang_stats[lang] = df.to_dict()

    # save lang_stats as a json
    with open(f'lang_stats.json', 'w') as f:
        json.dump(lang_stats, f, indent=4)

    save_all_lang_plots(lang_stats)


def save_features(all_mfcc_stats):
    for lang in all_mfcc_stats.keys():
        df = pd.DataFrame(all_mfcc_stats[lang])
        df.to_csv(f'{FEATURES_PATH}/{lang}.csv', index=False)


def load_data(langs):
    data = {}
    for lang in langs:
        data[lang] = []
        for file in os.listdir(f'{DATASET_PATH}/{lang}'):
            if file.endswith(".mp3"):
                data[lang].append(file)

    return data


def process_data(data, save_mfcc, num_spectrograms):
    all_mfcc_stats = { lang: [] for lang in data.keys() }

    for lang in data.keys():
        logging.info(f"Processing {lang} language")
        num_files = len(data[lang])
        current_file = 0
        sys.stdout.write("\rProcessed 0 % files")
        for file in data[lang]:
            try:
                mfccs, mfcc_stats = extract_mfcc(file, lang)
                mfcc_stats['file'] = file

                all_mfcc_stats[lang].append(mfcc_stats)
                
                if save_mfcc:
                    save_audio_mfcc(mfccs, lang, file)

                if num_spectrograms > 0 and np.random.rand() < 0.1:
                    make_save_spectrogram(mfccs, lang, file)
                    num_spectrograms -= 1

                current_file += 1
                if current_file % 1000 == 0:
                    logging.info(f"Processed {current_file}/{num_files} files")

            except Exception as e:
                logging.error(f"Error processing file {file} in language {lang}: {e}")

        print()

        logging.info(f"Processed {lang} language\n")


    logging.info("Saving features extracted from audio files")
    save_features(all_mfcc_stats)

    logging.info("Saving language specific stats")
    save_all_lang_stats(all_mfcc_stats)
        

def main(languages, save_mfcc, num_spectrograms):

    logging.info("Loading data")
    data = load_data(languages)
    logging.info("Data loaded\n")

    logging.info("Starting data processing\n")
    process_data(data, save_mfcc, num_spectrograms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MFCC features from audio files')
    parser.add_argument('--lang', type=str, nargs='+', help='list of languages', default='all')
    parser.add_argument('--save_mfcc', type=bool, help='save mfcc features', default=False)
    parser.add_argument('--num_spectrograms', type=int, help='number of spectrograms to save', default=0)

    args = parser.parse_args()
    
    languages = args.lang

    if languages == "all":
        languages = os.listdir(DATASET_PATH)

    if not os.path.exists(SPECTROGRAM_PATH):
        os.makedirs(SPECTROGRAM_PATH)
        for lang in languages:
            os.makedirs(f'{SPECTROGRAM_PATH}/{lang}')
    if not os.path.exists(MFCC_PATH):
        os.makedirs(MFCC_PATH)
        for lang in languages:
            os.makedirs(f'{MFCC_PATH}/{lang}')
    if not os.path.exists(FEATURES_PATH):
        os.makedirs(FEATURES_PATH)
    if not os.path.exists(LANG_INFO_PLOT_PATH):
        os.makedirs(LANG_INFO_PLOT_PATH)

    logging.basicConfig(level=logging.INFO, filename='logs/processing.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')
    
    logging.info("AUDIO PROCESSING - MFCC FEATURES EXTRACTION")
    logging.info(f"Languages selected:\n{languages}\n")
    logging.info(f"Save MFCC features: {args.save_mfcc}")
    logging.info(f"Number of spectrograms to save: {args.num_spectrograms}\n")

    main(languages, args.save_mfcc, args.num_spectrograms)
    