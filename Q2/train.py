import argparse
import logging
import os
from train_utils.train_nn import main as train_nn_main
from train_utils.train_ml import main as train_ml_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename=f'logs/train_{args.model}.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filemode='w')

    if args.model == 'nn':
        train_nn_main(args.epochs)
    elif args.model in ['svm', 'knn', 'rf', 'dt']:
        train_ml_main(args.model)
    else:
        logging.error('Invalid model')
        exit(1)