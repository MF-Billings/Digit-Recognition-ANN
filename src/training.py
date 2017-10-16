import logging
import config
import ann_io
from my_math import *

# activation functions
logger = logging.getLogger(__name__)


# threshold determines how close an output needs to be to the desired output to be classified as a yes or no
class Trainer:
    def __init__(self):
        self.avg_output_error = {       # each element is the error for a run
            "training": [],
            "validation": [],
            "testing": []
        }
        self.correct_outputs = {
            "training": [],
            "validation": [],
            "testing": []
        }
        self.input_size = {
            "training": -1,
            "validation": -1,
            "testing": -1
        }

    # assume desired output range is [0,1] for now
    @staticmethod
    def output_is_correct(true_output, output, o_max=1, o_min=0):
        # a yes
        mean = float(o_max + o_min) / 2
        if mean < output <= true_output:
            return True
        else:
            return False

    def get_results(self, data_set):
        results = []
        n_records = len(self.avg_output_error[data_set])
        # provide field names
        results.append("Data Set,Correct Classifications,Input Size,Average Output Error")
        for i in xrange(n_records):
            results.append("%s,%d,%d,%f" % (data_set, self.correct_outputs[data_set][i], self.input_size[data_set], self.avg_output_error[data_set][i]))
        return results

    # using k-fold validation

    def run(self, ann, training_data, max_epochs, epsilon, activation_fn=sigmoid, k=0):
        # holdout
        if k == 0:
            # split input set into training and validation sets
            partitions = ann_io.partition_sets(training_data, config.random, .2, .05)
            training_set = partitions[0]
            validation_set = partitions[1]

            self.__training_block(ann, training_set, max_epochs, epsilon, validation_set, activation_fn)

        # k fold
        else:
            # NOTE this isn't set up to handle numbers that k isn't a factor of
            split_size = len(training_data)/k

            # continue training until improvement between epochs is negligible (whatever that may mean)
            for f in xrange(k):
                training_set = training_data[0: f*split_size]
                validation_set = training_data[f*split_size:(f+1)*split_size]
                training_set.extend(training_data[(f+1)*split_size+1: len(training_data)])
                self.__training_block(ann, training_set, max_epochs, epsilon, validation_set, activation_fn)


    def __training_block(self, ann, training_set, max_epochs, epsilon, validation_set, activation_fn=sigmoid):
        prev_validation_error = float("inf")
        avg_validation_error = 0
        i = 0

        # continue training until improvement between epochs is negligible (whatever that may mean)
        while i < max_epochs : # and (avg_validation_error + epsilon) < prev_validation_error
            logging.info(
                "Printing training results...Order of data for each row printed is [epoch iteration digit expected actual total_error]")

            if i > 0:
                prev_validation_error = avg_validation_error
            # shuffle to discourage memorization
            ann_io.shuffle_list(config.random, training_set)
            # 1 epoch
            print "TRAINING================================================================================================"
            # the desired output data
            total_set_error = 0
            correct_outputs = 0
            for j in xrange(len(training_set)):
                ann.inputs = np.array(training_set[j].obj)
                ann.expected_outputs = np.array(
                    [1 if training_set[j].expected_value == k else 0 for k in xrange(len(ann.outputs))])
                ann.forward(activation_fn)

                if i == j == 0:
                    logger.info("before training: %f" % ann.e_sum_output)
                    for k in xrange(len(ann.outputs)):
                        logger.info("output %d: expected: %f, actual %f" % (k, ann.expected_outputs[k], ann.outputs[k]))
                else:
                    for k in xrange(len(ann.outputs)):
                        # the index with a 1 represents the digit expected
                        if ann.expected_outputs[k] == 1:
                            logger.info("%d %d %d %f %f %f" % (
                                i, j, k, ann.expected_outputs[k], ann.outputs[k], ann.e_sum_output))
                            if self.output_is_correct(ann.expected_outputs[k], ann.outputs[k]):
                                correct_outputs += 1
                    total_set_error += ann.e_sum_output

                ann.backward(activation_fn, config.quickprop)

            # record epoch results
            self.input_size["training"] = len(training_set)
            avg_set_error = total_set_error / len(training_set)
            self.avg_output_error["training"].append(avg_set_error)
            self.correct_outputs["training"].append(correct_outputs)

            self.validate(ann, validation_set, activation_fn)

            print "EPOCH RESULTS======================================================================================="
            # check for over-training with validation set
            # don't think I need to shuffle the inputs as I'm not changing the weights
            avg_validation_error = self.avg_output_error["validation"][len(self.avg_output_error["validation"]) - 1]
            correct_op = self.correct_outputs["validation"][len(self.correct_outputs["validation"]) - 1]
            print "%d/%d or %f of the input patterns correctly classified" % (
                correct_op,
                len(validation_set),
                (correct_op / (len(validation_set) * 1.0)))
            i += 1

        logger.info("TRAINING COMPLETE=================================================================================")
        logger.info("total output error after %d epochs: %f" % (i, ann.e_sum_output))
        logger.info("==================================================================================================")

    # contains common code for validation and testing
    def __testing_block(self, ann, input_set, input_type, activation_fn=sigmoid):
        # the desired output data
        total_set_error = 0
        correct_outputs = 0

        for j in xrange(len(input_set)):
            ann.inputs = np.array(input_set[j].obj)
            ann.expected_outputs = np.array(
                [1 if input_set[j].expected_value == k else 0 for k in xrange(len(ann.outputs))])
            ann.forward(activation_fn)

            for k in xrange(len(ann.outputs)):
                # k serves as the digit expected
                if ann.expected_outputs[k] == 1:
                    print "%d,%d,%s,%f,%f,%f" % (j, k, input_type, ann.expected_outputs[k], ann.outputs[k], ann.e_sum_output)
                    if self.output_is_correct(ann.expected_outputs[k], ann.outputs[k]):
                        correct_outputs += 1
            total_set_error += ann.e_sum_output
        # record epoch results
        self.input_size[input_type] = len(input_set)
        avg_set_error = total_set_error / len(input_set)
        self.avg_output_error[input_type].append(avg_set_error)
        self.correct_outputs[input_type].append(correct_outputs)

    # test the ANN and decide when to stop training
    # returns the error, correct outputs for each epoch.  There should be improvement between training sessions.
    def validate(self, ann, validation_set, activation_fn=sigmoid):
        logger.info(
            "VALIDATION===============================================================================================")
        self.__testing_block(ann, validation_set, "validation", activation_fn)

    def test(self, ann, testing_set, activation_fn=sigmoid):
        logger.info(
            "TESTING==================================================================================================")
        self.__testing_block(ann, testing_set, "testing", activation_fn)

        correct_op = self.correct_outputs["testing"][len(self.correct_outputs["testing"])-1]
        print "%d/%d or %f of the input patterns correctly classified" % (
            correct_op,
            len(testing_set),
            (correct_op / (len(testing_set) * 1.0)))
