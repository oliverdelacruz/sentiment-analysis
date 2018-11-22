########################################################################################################################
#WORD2VEC MODEL
########################################################################################################################
# Import libaries
from sklearn.manifold import TSNE
from random import shuffle
import time
import datetime
import pickle
import collections
import math
import os
import random
import numpy as np
import tensorflow as tf
import pandas

# Import config file
from config import FLAGS

class Word2vec():
    def __init__(self, embedding_size = 100, skip_window = 5, num_skips = 2, max_vocab_size= 80000,
                 min_occurrences = 5, batch_size = 128, learning_rate= 0.05, num_sampled = 3,
                 sample =  1e-4, dir_vocab = None, dict_words = None):
        self.embedding_size = embedding_size
        self.vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cooccurrence_matrix = None
        self.embeddings = None
        self.dir_vocab = dir_vocab
        self.word_to_id = None
        self.data_index = 0
        self.skip_window = self.left_context = self.right_context = skip_window
        self.num_skips = num_skips
        self.num_sampled =  num_sampled * self.batch_size
        self.sample = sample
        self.word_to_id = dict_words
        self.words = list(dict_words.keys())
        self.prob_vocab = {}
        print(self.words)
        print(len(self.words))
        print(self.words[:10])

        # Load vocabulary
        if self.dir_vocab != None:
            with open(self.dir_vocab, 'rb') as f:
                self.word_to_id = pickle.load(f)
                self.words = sorted(self.word_to_id, key=self.word_to_id.get)

        # Output directory
        self.timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.timestamp))
        os.makedirs(self.out_dir)

    def fit_to_corpus(self, corpus):
        # Load data and build the graph
        self.compute_prob(corpus)
        self.build_sampling(corpus)
        self.build_graph()

    def compute_prob(self,corpus):
        # Count per word
        counter = collections.Counter([item for sublist in corpus for item in sublist])
        special_tab_counter = counter.most_common(3)
        threshold_count = (sum(counter.values()) - sum(dict(special_tab_counter).values())) * self.sample
        for w in counter.keys():
            v = counter[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability > 1.0:
                word_probability = 1.0
            self.prob_vocab[w]  = word_probability
        result = pandas.DataFrame(np.transpose(np.vstack((np.array(list(self.prob_vocab.keys())), np.array(list(self.prob_vocab.values()))))))
        result.to_csv("prob.csv")

    def build_sampling(self, corpus):
        batch = []
        labels = []
        # Reorder the words by frequency
        # Process each sentence and perform subsampling
        j = 0
        start = time.time()
        end = time.time()
        print("Time counting: {}".format(str(end - start)))
        for sentence in corpus:
            word_vocabs = [self.word_to_id[w] for w in sentence if self.prob_vocab[w] > np.random.rand()]
            right_size = left_size = np.random.randint(self.skip_window, size = len(word_vocabs)) + 1
            for l_context, word, r_context in context_windows(word_vocabs, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    labels.append(context_word)
                    batch.append(word)
                for i, context_word in enumerate(r_context):
                    labels.append(context_word)
                    batch.append(word)
            if j % 100000 == 0:
                end = time.time()
                print("Time counting: {}".format(str(end - start)))
                print(j)
                start = time.time()
            j +=1

        # Save output
        self.labels = np.reshape(np.array(labels),(np.array(labels).shape[0], 1))
        self.inputs = np.array(batch)

    def create_batches(self, labels, inputs):
        # Calculate number of batches
        self.num_batches = int(labels.shape[0] / self.batch_size) + 1

        # Add random sentence to complete batches
        num_add_integers = ((self.num_batches)  * self.batch_size) - labels.shape[0]
        rand_indices = np.random.choice(labels.shape[0],num_add_integers)
        self.labels = np.vstack((labels, labels[rand_indices,:]))
        print(self.labels.shape)
        self.inputs = np.hstack((inputs, inputs[rand_indices])).flatten()
        print(self.labels.shape)
        print(self.inputs.shape)

        # Split data into batches
        self.labels = np.split(self.labels, self.num_batches)
        self.inputs = np.split(self.inputs, self.num_batches)

    # Step 3: Function to generate a training batch for the skip-gram model.
    def generate_batch(self,batch_size, num_skips, skip_window):
        # Assert conditions
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        # Initialize arrays
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        # For loop to get the data
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)

        # For loop to sample context given a world
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
          # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set random seed
            tf.set_random_seed(13)
            # Define inputs
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            # Define labels
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size,1])

            # Look up embeddings for inputs.
            self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0
            / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute the average NCE loss for the batch.
            self.total_loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases,labels = self.train_labels,
            inputs = embed, num_sampled = self.num_sampled, num_classes = self.vocab_size), name = "sum_losses")

            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.total_loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm

    def train(self):
         # Lauch tensorflow session
        with tf.Session(graph=self.graph) as session:
            # Output directory for models and summaries
            print("Writing to {}\n".format(self.out_dir))
            total_steps = 0

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", self.total_loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

            # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
            checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

            # Initialize variables
            tf.global_variables_initializer().run()
            session.graph.finalize()

            # Generate batches
            self.create_batches(self.labels, self.inputs)

            # Run training procedure
            for epoch in range(FLAGS.num_epochs):
                # Shuffle
                perm_idx = np.random.permutation(len(self.labels)).tolist()
                for batch_index in perm_idx:
                    # Define feed dictionary
                    feed_dict = {self.train_inputs: self.inputs[batch_index],
                    self.train_labels: self.labels[batch_index]}
                    # Run optimization
                    _, mean_loss, summaries = session.run([self.optimizer, self.total_loss, train_summary_op],
                    feed_dict=feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    if total_steps % FLAGS.evaluate_every == 0:
                        train_summary_writer.add_summary(summaries, total_steps)
                        print("{}: step {}, loss {:g}".format(time_str, total_steps, mean_loss))
                    if total_steps % FLAGS.checkpoint_every == 0:
                        path = saver.save(session, checkpoint_prefix, global_step=total_steps)
                        print("Saved model checkpoint to {}\n".format(path))
                    total_steps += 1
                current_embeddings = self.embeddings.eval()
                current_embeddings_normalized = self.normalized_embeddings.eval()
                output_path = os.path.join(self.out_dir, "epoch{:03d}.png".format(epoch + 1))
                self.generate_tsne(output_path, embeddings = current_embeddings)
                # Save output
                np.save(os.path.join(self.out_dir, "embeddings_word2vec"), current_embeddings)
                np.save(os.path.join(self.out_dir, "embeddings_word2vec_normalized"), current_embeddings_normalized)

    def generate_tsne(self, path=None, size=(100, 100), word_count=2500, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)

def context_windows(region, left_size_window, right_size_window):
    for i, word in enumerate(region):
        if left_size_window.shape[0] > 1 or right_size_window.shape[0] > 1:
            left_size = left_size_window[i]
            right_size = right_size_window[i]
        else:
            left_size = left_size_window
            right_size = right_size_window
        start_index = i - left_size
        end_index = i + right_size
        left_context = window(region, start_index, i - 1)
        right_context = window(region, i + 1, end_index)
        yield (left_context, word, right_context)

def window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) - 1
    selected_tokens = region[max(start_index, 0): (min(end_index, last_index) + 1)]
    return selected_tokens
