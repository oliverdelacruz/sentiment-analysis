########################################################################################################################
#GLOVE MODEL
########################################################################################################################
# Import libaries
import os
import tensorflow as tf
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from random import shuffle
import time
import datetime
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Import config file
class GloVeModel():
    def __init__(self,min_occurrences = 5,
                 scaling_factor = 3/4, cooccurrence_cap = 100, learning_rate= 0.05,
                 dict_vocab = None,  config = None):
        self.window_size = config.window_size/2
        if isinstance(self.window_size, int):
            self.left_context = self.right_context = self.window_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.learning_rate = learning_rate
        self.cooccurrence_matrix = None
        self.embeddings = None
        self.dict_vocab = dict_vocab
        self.word_to_id = None

        # Parameters defined in configuration file
        self.config = config
        self.out_dir = config.output_dir
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size + config.vocab_tags
        self.debug = config.debug
        self.grad_clip = config.grad_clip
        self.allow_soft_placement = config.allow_soft_placement
        self.log_device_placement = config.log_device_placement
        self.num_checkpoint = config.n_checkpoints_to_keep
        self.num_epoch = config.n_epochs
        self.save_every = config.save_every
        self.evaluate_every = config.evaluate_every
        self.learning_rate_decay_factor = config.decay_learning_rate
        self.keep_prob_dropout = config.dropout_prob_keep
        self.data_dir = config.data_dir
        self.vocab_path_file = config.vocab_file
        self.sentence_path_file = config.tweets_file

        # Special vocabulary symbols
        self.pad = r"_PAD"
        self.bos = r"_BOS"
        self.eos = r"_EOS"
        self.unk = r"_UNK"

        # Load vocabulary
        if self.dict_vocab != None:
                self.word_to_id = self.dict_vocab
                self.words = list(self.word_to_id.keys())
        else:
            self.words = None

        # Output directory
        self.timestamp = str(int(time.time()))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.timestamp))
        os.makedirs(self.out_dir)

    def fit_to_corpus(self, bool_processing = True):
        # Load tweets
        path_tweets = os.path.join(self.data_dir, self.sentence_path_file)
        with open(path_tweets, 'rb') as f:
            corpus = pickle.load(f)

        # Conversion from integers to words
        corpus = self.add_tags(corpus)

        # Build cooccurence matrix
        self.build_cooc_matrix(corpus, self.vocab_size, self.min_occurrences, self.left_context, self.right_context)

        # Build graph
        self.build_graph()

    def add_tags(self, corpus, bool_add_tags = False):
        # Convert integers to words
        tokens = []
        if bool_add_tags:
            for line in corpus:
                # Add special tags
                line.append(self.eos)
                line.insert(0, self.bos)
                tokens.append(line)
            return tokens
        else:
            return corpus

    def build_cooc_matrix(self, corpus, vocab_size, min_occurrences, left_size, right_size):
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        for line in corpus:
            word_counts.update(line)
            for l_context, word, r_context in context_windows(line, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No cooccurrences in corpus. Did you try to reuse a generator?")
        self.cooccurrence_matrix = {
            (self.word_to_id[words[0]], self.word_to_id[words[1]]): count
            for words, count in cooccurrence_counts.items()
            if words[0] in self.word_to_id and words[1] in self.word_to_id}
        print("Saving cooccurence matrix")
        with open(os.path.join(self.out_dir, "dict_cooc.pkl") , 'wb') as f:
            pickle.dump(self.cooccurrence_matrix, f)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/gpu:0'):
                # Set random seed
                tf.set_random_seed(13)
                # Limit the maximum number of counts
                count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32, name='max_cooccurrence_count')
                # Define scaling factor
                scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32, name="scaling_factor")
                # Define matrix words (int)
                self.local_input = tf.placeholder(tf.int32, shape=[self.batch_size], name="local_words")
                # Define the context matrix words (int)
                self.context_input = tf.placeholder(tf.int32, shape=[self.batch_size], name="context_words")
                # Define the concurrence matrix (counts)
                self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size], name="cooccurrence_count")
                # Define local embeddings
                local_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], 1.0, -1.0),
                name="local_embeddings")
                # Define contect embeddings
                context_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim ], 1.0, -1.0),
                name="context_embeddings")
                # Define local bias
                local_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name="local_biases")
                # Define context bias
                context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0), name="context_biases")

                # Lookup table for embeddings
                local_embedding = tf.nn.embedding_lookup([local_embeddings], self.local_input)
                context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)

                # Lookup bias for embedding
                local_bias = tf.nn.embedding_lookup([local_biases], self.local_input)
                context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

                # Define weighting factor
                weighting_factor = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count, count_max), scaling_factor))

                # Matrix multiplication and perform log of counts for the cooccurence matrix
                embedding_product = tf.reduce_sum(tf.multiply(local_embedding, context_embedding), 1)
                log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))
                distance_expr = tf.square(tf.add_n([embedding_product, local_bias, context_bias, tf.negative(log_cooccurrences)]))
                single_losses = tf.multiply(weighting_factor, distance_expr, name = "mult_losses")

                # Set up the loss fuction
                self.total_loss = tf.reduce_sum(single_losses, name = "sum_losses")
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

                # Combine contenxt and local embeddings
                self.combined_embeddings = tf.add(local_embeddings, context_embeddings, name="combined_embeddings")

    def train(self):
        # Prepare batches
        batches = self.prepare_batches()
        total_steps = 0

        # Lauch tensorflow session
        session_conf = tf.ConfigProto(
        allow_soft_placement= self.allow_soft_placement,
        log_device_placement= self.log_device_placement,
        inter_op_parallelism_threads=36,
        intra_op_parallelism_threads=36,
        )
        with tf.Session(graph=self.graph,config=session_conf) as session:
            # Output directory for models and summaries
            print("Writing to {}\n".format(self.out_dir))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", self.total_loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(self.out_dir , "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

            # Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it)
            checkpoint_dir = os.path.abspath(os.path.join(self.out_dir , "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(max_to_keep = self.num_checkpoint)

            # Initialize variables
            tf.global_variables_initializer().run()
            session.graph.finalize()

            # Run training procedure
            for epoch in range(self.num_epoch * 20):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    feed_dict = {
                        self.local_input: i_s,
                        self.context_input: j_s,
                        self.cooccurrence_count: counts}
                    _, mean_loss, summaries = session.run([self.optimizer, self.total_loss, train_summary_op], feed_dict=feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    if total_steps % self.evaluate_every == 0:
                        train_summary_writer.add_summary(summaries, total_steps)
                        print("{}: step {}, loss {:g}".format(time_str, total_steps, mean_loss))
                    if total_steps % self.save_every * 100 == 0:
                        path = saver.save(session, checkpoint_prefix, global_step= total_steps)
                        print("Saved model checkpoint to {}\n".format(path))
                    total_steps += 1
                current_embeddings = self.combined_embeddings.eval()
                output_path = os.path.join(self.out_dir , "epoch{:03d}.png".format(epoch + 1))
                self.generate_tsne(output_path, embeddings= current_embeddings)
                # Save output
                np.save(os.path.join(self.out_dir, "embeddings_glove"), current_embeddings)

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def prepare_batches(self):
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(batchify(self.batch_size, i_indices, j_indices, counts))

    def generate_tsne(self, path=None, size=(100, 100), word_count=1500, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

def context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
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

def batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)

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

def embedding_search(self, word_str_or_id):
    if isinstance(word_str_or_id, str):
        return self.embeddings[self.word_to_id[word_str_or_id]]
    elif isinstance(word_str_or_id, int):
        return self.embeddings[word_str_or_id]
