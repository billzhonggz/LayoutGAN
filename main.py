"""Entrance of this LayoutGAN implementation.

This file handles command-line calls with arguments.

List of available arguments and options.
--train --dataset-path --output-model-path --number-of-epoch trigger a training process.
--evaluate --pretrained-model-path --output-data-path

©2019-current, Junru Zhong, all rights reserved.
"""

import sys
import getopt

long_args = [
    'help',
    'train',
    'dataset-path=',
    'output-model-path=',
    'number-of-epoch=',
    'evaluate',
    'pretrained-model-path=',
    'output-data-path='
]


def usage():
    print('Usage: \n' + sys.argv[0] +
          ' --train --dataset-path= --output-model-path= --number-of-epoch= trigger a training process.\n' +
          sys.argv[0] + ' --evaluate --pretrained-model-path= --output-data-path= to start a evaluation.')


def train(optlist):
    """Process arguments when training mode is triggered.
    Report errors when
    1. Missing arguments.
    2. Path provided is invalid.
    """
    dataset_path = ''
    output_model_path = ''
    number_of_epoch = ''
    for opt, arg in optlist:
        if opt == '--dataset-path':
            dataset_path = arg
        if opt == '--output-model-path':
            output_model_path = arg
        if opt == '--number-of-epoch':
            number_of_epoch = arg
    if dataset_path or output_model_path or number_of_epoch is '':
        print('Missing arguments for training.')
        usage()
        sys.exit(2)
    else:
        # TODO: Start a train.
        print(dataset_path + '\n' + output_model_path + '\n' + number_of_epoch)


def evaluate(optlist):
    """Process arguments when evaluation mode is triggered.
    Report errors when
    1. Missing arguments.
    2. Path provided is invalid.
    """
    pretrained_model_path = ''
    output_data_path = ''
    for opt, arg in optlist:
        if opt == '--pretrained-model-path':
            pretrained_model_path = arg
        if opt == '--output-model-path':
            output_data_path = arg
    if pretrained_model_path or output_data_path is '':
        print('Missing arguments for evaluation.')
        usage()
        sys.exit(2)
    else:
        # TODO: Start a evaluate.
        print(pretrained_model_path + '\n' + output_data_path)


def main():
    """The main function.
    Read the arguments, and direct the program accordingly.
    Report errors when any invalid argument is given.
    """
    try:
        (optlist, args) = getopt.getopt(sys.argv[1:], '', long_args)
        # print(optlist)
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    # Analysis the arguments and trigger functions.
    for opt, arg in optlist:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '--train':
            print('Enter training mode.')
            train(optlist)
        elif opt == '--evaluate':
            print('Enter evaluate mode.')


if __name__ == "__main__":
    main()
