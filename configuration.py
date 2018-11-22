# NLU Project
# Configuration file
# Description: The script setup all the parameters

# Import libraries
import tensorflow as tf
import os
import sys
import time
import csv
from pathlib import PurePath

# Define parent directory
project_dir = str(PurePath(__file__).parent)  # root of git-project

# Labels for output and data
label_output = "runs"
label_data = "data"
label_emb = "emb"

# Define output directory
timestamp = str(time.strftime('%d-%m-%H-%M-%S'))
output_dir = os.path.abspath(os.path.join(os.path.curdir, label_output, timestamp))

# Setup constant parameters
flags = tf.app.flags

# Define directory parameters and file names
flags.DEFINE_string('output_dir', output_dir, 'The directory where all results are stored')
flags.DEFINE_string('data_dir', os.path.join(project_dir, label_data), 'The directory where all input data are stored')
flags.DEFINE_string('emb_dir', os.path.join(project_dir, label_data, label_emb), 'The directory where all the embeddings are saved')
flags.DEFINE_string('file_dir', os.path.join(project_dir, r"./model-390630"), 'The directory where all input data are stored')
flags.DEFINE_string('vocab_file', "vocab.pkl", 'The name of the vocabulary file')
flags.DEFINE_string('tweets_file', "tweets.pkl", 'The name of the tweets file - all the sentences')

# Define model parameters
flags.DEFINE_bool('debug', True, 'Run in debug mode')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('rnn_size', 150, 'Number of hidden units')
flags.DEFINE_integer('rnn_size_reduced', 128, 'Number of hidden units')
flags.DEFINE_integer('embedding_dim', 100,'The dimension of the embedded vectors')
flags.DEFINE_string('model_name', 'seq2seq', 'Name of the trained model')
flags.DEFINE_integer('vocab_size', 88027, 'Total number of different words')
flags.DEFINE_float('grad_clip', 10, 'Limitation of the gradient')
flags.DEFINE_integer('max_seq_length', 60, 'Maximum sequence length')
flags.DEFINE_integer('vocab_tags', 4, 'Number of special tags')
flags.DEFINE_float('decay_learning_rate', 0.9, 'The decaying factor for learning rate')
flags.DEFINE_float('dropout_prob_keep', 0.75, 'The dropout probability to keep the units')
flags.DEFINE_float('n_units_attention', 128, 'The number of units for the attention')
flags.DEFINE_integer('window_size', 5, 'The number of units for the attention')
flags.DEFINE_integer('add_tags', 1, 'The number of units for the attention')
flags.DEFINE_string('emb_model', "word2vec" , 'The number of units for the attention')
flags.DEFINE_integer('iter', 120, 'The number of itinerations for word2vec')

# Define training parameters
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('n_epochs', 10, 'Number of epochs')
flags.DEFINE_string('mode', "train" , 'The number of units for the attention')

# Define general parameters
flags.DEFINE_integer('summary_every', 10, "generate a summary every `n` step. This is for visualization purposes")
flags.DEFINE_integer('n_checkpoints_to_keep', 10,'keep maximum `integer` of chekpoint files')
flags.DEFINE_integer('evaluate_every', 8000,'evaluate trained model every n step')
flags.DEFINE_integer('save_every', 20000, 'store checkpoint every n step')

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Obtain the current paremeters
def get_configuration():
    global FLAGS
    FLAGS = tf.flags.FLAGS
    FLAGS(sys.argv)
    return FLAGS

# Print the current paramets
def print_configuration():
    print("Parameters: ")
    filename = os.path.join(output_dir, "config.csv")
    if FLAGS.mode != "infer":
        os.makedirs(output_dir)
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for attr, value in sorted(FLAGS.__flags.items()):
                csvwriter.writerow([attr.upper(), value])
                print("{}={}".format(attr.upper(), value))
    else:
        for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(), value))
