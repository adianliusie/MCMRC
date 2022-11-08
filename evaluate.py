import argparse
import numpy as np

from transformers import logging

from src.handlers.system_loader import EnsembleLoader

# Define parser for arguments of the script
parser = argparse.ArgumentParser(description='Arguments for training the system')

parser.add_argument('--path',                      type=str,  help='name to save the experiment as')
parser.add_argument('--data-set', default='race',  type=str,  help='dataset to evaluate')
parser.add_argument('--mode',     default='test',  type=str,  help='dataset to evaluate')


if __name__ == '__main__':
    args = parser.parse_args()

    # remove warning of missing parameters in the checkpoint
    logging.set_verbosity_error() 
    
    # load system
    system = EnsembleLoader(args.path)

    # load probs (if calculated before, they are )
    probs = system.load_probs(args.data_set, args.mode)
    labels = system.load_labels(args.data_set, args.mode)

    # calculate and print accuracy
    accuracy = system.calc_accuracy(probs, labels)
    print(f'accuracy: {accuracy:.2f}')