# handle program I/O
import math
import logging
from input_unit import InputUnit

logger = logging.getLogger(__name__)


# do all the necessary work to format the input data so the ANN can process it
def format_data(filepaths, rand=None):
    inputs = merge_inputs(filepaths)
    if rand is not None:
        shuffle_list(rand, inputs)
    return inputs


def write_to_file(filepath, list):
    with open(filepath, "w") as f:
        for index in xrange(len(list)):
            f.write(list[index] + "\n")


# shuffles in place using a random object
def shuffle_list(rand, alist):
    logging.debug("Before shuffling training pattern")
    logging.debug(alist)
    rand.shuffle(alist)
    logging.debug("After shuffling training pattern")
    logging.debug(alist)


def read_digit_file(filepath):
    logging.info("Reading digit file %s", filepath)
    # 'with' closes file afterwards
    with open(filepath, 'r') as f:
        din = [tuple(map(float, line.split(','))) for line in f]
    logger.debug("Samples: %d", len(din))
    logger.debug("Patterns")
    for pattern in din:
        logger.debug(pattern)
    return din


def read_input_files(filepaths):
    input_data = {}

    for i in xrange(len(filepaths)):
        input_data[i] = read_digit_file(filepaths[i])
    return input_data


# consolidate individual inputs within the files into a single list
def merge_inputs(filepaths):
    input_data = read_input_files(filepaths)
    inputs = []

    for i in xrange(len(input_data)):
        for j in xrange(len(input_data[i])):
            inputs.append(InputUnit(input_data[i][j], i))
    return inputs


# Split the input data into a training and a validation set
# all set sizes given as percents
def partition_sets(inputs, rand, training_split=0.8, validation_split=0.2):
    training_size = math.floor(len(inputs) * training_split)
    validation_size = math.ceil(len(inputs) * validation_split)
    # testing_size = len(inputs) - training_size - validation_size
    shuffle_list(rand, inputs)
    offset = int(training_size)
    training_set = [inputs[t] for t in xrange(int(training_size))]
    validation_set = [inputs[v] for v in xrange(offset, offset + int(validation_size))]
    return training_set, validation_set
