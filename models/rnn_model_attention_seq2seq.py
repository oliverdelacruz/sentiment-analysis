# NLU Project
# Main file
# Description: The script creates a class for the model seq2seq

# Import site-packages
import tensorflow as tf
from models.basic_model import BasicModel
import numpy as np
import datetime
import pandas as pd

class RNN_Model_Attention(BasicModel):
    # Build the graph
    def _build_graph(self):
        # Set random seed
        tf.set_random_seed(13)

        # Allocate to GPU
        with tf.device('/gpu:0'):
            # Load inputs
            self.input = tf.placeholder(tf.int32, [None, None], name="input")

            # The length of the sentences
            self.input_length = tf.placeholder(tf.int32, [None], name="input_length")


            # The lenght of the decoder (dummy variable)
            self.decoder_length = tf.placeholder(tf.int32, [None], name="decoder_length")

            # Load labels
            self.labels = tf.placeholder(tf.int32, [None], name="labels")

            # Define additional parameters
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.learning_rate = tf.Variable(self.learning_rate, trainable=False, dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)

            # Add drouput
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            # Load labels to compute loss
            self.max_sequence = tf.reduce_max(self.input_length, name = "max_sequence")

            # Load embedding
            emb_initial = tf.constant(self.emb, dtype=tf.float32)
            emb_pretrained = tf.Variable(emb_initial[4:, :], trainable=False)
            emb_train = tf.Variable(tf.random_uniform([4, self.embedding_dim], -0.1, 0.1))
            emb = tf.concat([emb_train, emb_pretrained], axis=0, name="emb")
            print('Printing embeddings tensor size: {}'.format(str(emb.get_shape())))


            # Define constants
            self.const_input = tf.constant(np.ones([self.batch_size * 2]) * 2, dtype=tf.int32)

            # Create a vector for end of sentence used as input for decoder
            self.input_decoder = tf.gather(self.const_input, self.input_length - 1, name="input_decoder")

            # Embedding layer
            with tf.name_scope("embedding"):
                emb_words_encoder = tf.nn.embedding_lookup(emb, self.input[:,:self.max_sequence - 1])
                emb_words_decoder = tf.reshape(tf.nn.embedding_lookup(emb, self.input_decoder),[-1,1,self.embedding_dim])
            print('Printing input tensor size: {}'.format(str(self.input.get_shape())))
            print('Printing embeddings tensor size: {}'.format(str(emb_words_encoder.get_shape())))
            print('Printing embeddings decoder tensor size: {}'.format(str(emb_words_decoder.get_shape())))

            # Rnn layer - encoder
            with tf.variable_scope("rnn-attention") as scope:
                # Train - encoder
                lstm_encoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size),
                                                             output_keep_prob = self.keep_prob)
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(lstm_encoder, emb_words_encoder,
                                                            sequence_length = self.input_length, dtype=tf.float32)
                print('Printing outputs tensor size: {}'.format(str(encoder_outputs.get_shape())))

                # Attention mechanism
                lstm_decoder = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.rnn_size),
                                                             output_keep_prob=self.keep_prob)
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.rnn_size, encoder_outputs,
                    name='LuongAttention')
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_decoder,
                                                                attention_mechanism,
                                                                attention_layer_size=self.rnn_size,
                                                                name="AttentionWrapper")
                batch_size = tf.shape(self.input)[0]
                attn_zero = attn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                init_state = attn_zero.clone(cell_state=encoder_state)


                # Train - decoding
                helper_train = tf.contrib.seq2seq.TrainingHelper(inputs=emb_words_decoder, sequence_length=self.decoder_length)
                decoder_train = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell, helper=helper_train, initial_state=init_state)
                self.decoder_outputs, self.decoder_state, final_sequence_length = \
                tf.contrib.seq2seq.dynamic_decode(decoder=decoder_train,
                                                  output_time_major=False,
                                                  impute_finished=True)

            # Fully connected layer
            with tf.name_scope("rnn_layer"):
                # Parameters to calibrate
                W = tf.get_variable("W", [self.rnn_size, self.output_size], tf.float32,initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", [self.output_size], tf.float32, initializer=tf.zeros_initializer())

                # Calculate logits and predictions
                print(self.decoder_outputs.rnn_output.get_shape())
                logits = tf.nn.xw_plus_b(tf.reshape(self.decoder_outputs.rnn_output,
                                                    [-1, self.rnn_size]), W, b, name ="mul_logits")
                self.predictions = tf.argmax(logits, 1, name ="predictions")
                print('Printing logits tensor size: {}'.format(str(logits.get_shape())))
                print('Printing predictions tensor size: {}'.format(str(self.predictions.get_shape())))

            # Soft-max layer
            with tf.name_scope("softmax"):
                self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
                self.loss = tf.reduce_mean(self.losses, name="mean_loss")

            # Calculate accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(tf.reshape(tf.cast(self.predictions, tf.int32) , [-1]),
                                               self.labels, name="equal_predictions")
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="sum_accuracy")

            # Optimizer
            tvars = tf.trainable_variables()
            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads, tvars = zip(*optimizer.compute_gradients(self.loss, tvars))
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        # Call summary builder
        self._build_summary()

        # Call checkpoint builder
        self._save()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())
        self.session.graph.finalize()

        # Calculate number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.5fM" % (total_parameters / 1e6))

    def train(self):
        # Set up numpy seed
        np.random.seed(13)
        self.cum_loss = []

        # Train the model
        for n_epoch in range(self.num_epoch):
            # Create batches and shuffle
            self.avg_loss = []
            train_input, train_length_input, train_labels = \
                self.batchify(self.train_input, self.train_length_input, self.train_labels)

            # Train each batch
            for idx in range(len(train_input)):
                feed_dict = {self.input: train_input[idx],
                             self.input_length: train_length_input[idx],
                             self.labels: train_labels[idx,:],
                             self.decoder_length: np.ones((train_input[idx].shape[0],)),
                             self.keep_prob: self.keep_prob_dropout}
                _, step, summaries, loss_batch, accuracy_batch, = self.session.run(
                    [self.train_op, self.global_step, self.train_summary_op, self.loss, self.accuracy], feed_dict)
                if step % self.summary_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_batch, accuracy_batch))
                    self.train_summary_writer.add_summary(summaries, step)
                if step % self.save_every == 0:
                    path = self.saver.save(self.session, self.checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                if step % self.evaluate_every == 0:
                    self.avg_loss.append(np.mean(np.array(self.cum_loss).flatten()))
                    print("Average loss: {}".format(self.avg_loss[-1]))
                    if len(self.avg_loss) > 3 and n_epoch > 1 and max(self.avg_loss[-3:-1]) < self.avg_loss[-1]:
                        self.session.run(self.learning_rate_decay_op)
                        print("Changing the learning rate")
                    self.cum_loss = []
                self.cum_loss.append(loss_batch)

            # Save per epoch
            path = self.saver.save(self.session, self.checkpoint_prefix, global_step=step)
            print("Saved model checkpoint to {}\n".format(path))

    def infer(self):
        # Set up the saver
        saver = tf.train.import_meta_graph("{}.meta".format(self.file_dir))

        # Load the saved meta graph and restore variables
        saver.restore(self.session, self.file_dir)
        print(self.graph.get_operations())
        print(self.graph.get_all_collection_keys())
        for v in tf.global_variables():
            print(v.name)
            print(v.get_shape())
        print("Graph restored and calculating perplexity:")

        # Load operations
        self.input = self.graph.get_operation_by_name("input").outputs[0]
        self.input_length = self.graph.get_operation_by_name("input_length").outputs[0]
        self.labels = self.graph.get_operation_by_name("labels").outputs[0]
        self.keep_prob = self.graph.get_operation_by_name("keep_prob").outputs[0]
        self.predictions = self.graph.get_operation_by_name("rnn_layer/predictions").outputs[0]

        # Compute predictions
        pred = []
        for idx in range(len(self.test_input)):
            # Setup the feed dictionary
            feed_dict = {self.input: self.test_input[idx],
                         self.input_length: self.test_length_input[idx].flatten(),
                         self.labels: np.ones((self.test_input[idx].shape[0])),
                         self.keep_prob: 1.0}

            # Run operation
            pred.extend(self.session.run(self.predictions, feed_dict))

        # Prepare data for submission
        pred = np.array([np.arange(1,len(pred)+1).tolist(), pred]).T
        pred = pd.DataFrame(pred).replace(to_replace=[0, "0"], value=-1)
        filename = "submission.csv"
        header = ["Id", "Prediction"]
        pd.DataFrame(pred).to_csv(filename, header = header, index = False)
