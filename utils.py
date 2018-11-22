# NLU Project
# Description: The script performs the preprocessing of all the data

# Import site-packages libraries
import os
import re
import collections
import pandas
import pickle
import numpy as np
import pandas as pd
from shutil import copy2

# Import local modules from the package
from configuration import get_configuration

class Preprocessing():
    def __init__(self,config, train_path_file = ["train_neg_full.txt", "train_pos_full.txt"],
                 test_path_file =["test_data.txt"],
                 train_path_file_target ="input_train",
                 test_path_file_target ="input_test"):
        """Constructor: it initilizes the attributes of the class by getting the parameters from the config file"""
        self.train_path_file = train_path_file
        self.test_path_file = test_path_file
        self.train_path_file_target = train_path_file_target
        self.test_path_file_target = test_path_file_target
        self.data_dir = config.data_dir
        self.vocab_path_file = config.vocab_file
        self.sentence_path_file = config.tweets_file
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length
        self.add_tags = config.add_tags
        self.output_dir = config.output_dir

        # Special vocabulary symbols - we always put them at the start.
        self.pad = r"_PAD"
        self.bos = r"_BOS"
        self.eos = r"_EOS"
        self.unk = r"_UNK"
        self.start_vocab = [self.pad, self.bos, self.eos, self.unk]

        # Regular expressions used to tokenize.
        self.word_split = re.compile(r"([.,!?\"\';])")

        self.word_re = re.compile(r"<url>")
        self.word_sub = re.compile(r"^[0-9]*[0-9]*[0-9]*[0-9]*[0-9][,]*|^[0-9][,]")
        self.pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

        # Set up the paths
        self.path_vocab = os.path.join(self.data_dir, self.vocab_path_file)
        self.path_tweets = os.path.join(self.data_dir, self.sentence_path_file)

    def tokenizer(self, sentence, bool_tags = False):
        """Function: Very basic tokenizer: split the sentence into a list of tokens.
        Args:
            sentence: A line from the original file data.
        """
        # Split sentence by white spaces
        sentence = [self.word_split.split(self.pattern.sub(r"\1\1",self.word_re.sub(r"",word)))
                    for word in sentence.lower().strip().split()]

        # Flatten list
        sentence = [item for sublist in sentence for item in sublist]

        # Perform optional preprocessing
        sentence = [re.sub('^#+[a-z0-9,-_%!/\'?#@&()^+={}<>$|\"]*', '<hashtag>', word) for word in sentence]
        sentence = [re.sub('^[mbhae]+[uhaew]+[hae]+[hae]+$', '<haha>', word) for word in sentence]

        # Remove patterns very long sentences
        if len(sentence) >  self.max_seq_length :
            sentence = [word for word in sentence if not sentence.count(word) > 8]

        # Add special tags (bos & eos)
        if bool_tags:
            sentence.insert(0, self.bos)
            sentence.append(self.eos)

        # Remove empty lists
        sentence = [word for  word in sentence if word]

        # Output final sentence
        return sentence

    def create_vocabulary(self, input_path_file):
        """Function: Create vocabulary file (if it does not exist yet) from data file.
        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
        Args:
          input_path_file: data file that will be used to create vocabulary.
          tokenizer: a function to use to tokenize each data sentence; if None, internal tokenizer will be used.
        """

        # Check for an existing file
        if not os.path.exists(self.path_vocab):
            print("Creating vocabulary %s from data %s" % (self.vocab_path_file, self.data_dir))
            # Initialize dict and list
            self.vocab = {}
            tokens = []
            sentences = []
            counter = 0
            for file in input_path_file:
                path_input = os.path.join(self.data_dir, file)
                # Open the file
                with open(path_input, 'r', newline="\n", encoding='utf8') as f:
                    for line in f:
                        counter += 1
                        if counter % 50000 == 0:
                            print("Processing line %d" % counter)
                        # Process each line
                        line = self.tokenizer(line)[:]
                        tokens.extend(line)
                        sentences.append(line)

            # Generate dictionary by selecting the most common words
            counter = collections.Counter(tokens).most_common(self.vocab_size)

            # Save data for better visualization
            pandas.DataFrame.from_dict(counter).to_csv(os.path.join(self.data_dir, "vocab.csv"))

            # Create list of all the words in the vocabulary with the special tag
            self.vocab = dict(counter)
            vocab_list = self.start_vocab + sorted(self.vocab, key=self.vocab.get, reverse = True)

            # Save vocabulary
            print("Saving vocabulary")
            with open(self.path_vocab, 'wb') as f:
                pickle.dump(vocab_list, f)

            # Save all tokenized sentences
            print("Saving tweets")
            with open(self.path_tweets , 'wb') as f:
                pickle.dump(sentences, f)

    def initialize_vocabulary(self):
        """Function: Initialize vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
          dog
          cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].
        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).
        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        path_vocab = os.path.join(self.data_dir, self.vocab_path_file)
        if os.path.exists(path_vocab):
            with open(os.path.join(path_vocab), 'rb') as f:
                list_vocab = pickle.load(f)
            self.dict_vocab_reverse = dict([(idx, word) for (idx, word) in enumerate(list_vocab)])
            self.dict_vocab = dict((word, idx) for idx, word in self.dict_vocab_reverse.items())
        else:
            raise ValueError("Vocabulary file %s not found.", path_vocab)

    def sentence_to_token_ids(self, sentence, tokenizer=None):
        """Function: Convert a string to list of integers representing token-ids.
        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
        Args:
          sentence: the sentence in string format to convert to token-ids.
          tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        Returns:
          a list of integers, the token-ids for the sentence.
        """
        # Select tokenizer
        sentences = self.tokenizer(sentence, self.add_tags)

        # Convert to integers
        return [self.dict_vocab.get(word, self.dict_vocab.get(self.unk)) for word in sentences]

    def data_to_token_ids(self, input_path, target_path, train = True):
        """Tokenize data file and turn into token-ids using given vocabulary file.
        This function loads data line-by-line from data_path, calls the above
        sentence_to_token_ids, and saves the result to target_path. See comment
        for sentence_to_token_ids on the details of token-ids format.
        Args:
          data_path: path to the data file in one-sentence-per-line format.
          target_path: path where the file with token-ids will be created.
          vocabulary_path: path to the vocabulary file.
        """
        # Set up the path
        path_target = os.path.join(self.data_dir, target_path)

        # Initialize list
        tokens_ids = []
        tokens_length = []
        labels = []

        # Tokenize
        print("Tokenizing data in %s" % path_target)
        self.initialize_vocabulary()
        counter = 0
        for file in input_path:
            path_input = os.path.join(self.data_dir, file)
            with open(path_input, 'r', newline="\n", encoding='utf8') as f:
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("Tokenizing line %d" % counter)
                    if not train:
                        line = self.word_sub.sub(r"", line)
                    tokens_ids.append(self.sentence_to_token_ids(line))
                    tokens_length.append(len(tokens_ids[-1]))
                    # Insert labels for classification
                    if "pos" in file:
                        labels.append(1)
                    elif "neg" in file:
                        labels.append(0)

        # Print statistics
        print("Maximum length {}".format(max(tokens_length)))
        print("Average length {}".format(sum(tokens_length)/len(tokens_length)))
        print("Number of sentences {}".format(len(tokens_length)))
        n_unks = sum([tokens.count(3) for tokens in tokens_ids])
        n_words = sum([len(tokens) for tokens in tokens_ids])
        print("Number of unks {}".format(n_unks))
        print("Number of words {}".format(n_words))
        print("Ratio unks/words {}%".format(n_unks/n_words*100))

        # Print longest sentences
        np_tokenls_length = np.array(tokens_length)
        idx = np.argsort(np_tokenls_length)[-10:]
        for i in idx:
            print([self.dict_vocab_reverse.get(id) for id in tokens_ids[i]])

        return tokens_ids, tokens_length, labels

    def prepare_data(self):
        """Prepare all necessary files that are required for the training.
          Args:
          Returns:
            A tuple of 2 elements:
              (1) list of the numpy token-ids for training data-set
              (2) list of the numpy token-ids for test data-set,
          """
        # Set up the path
        self.path_target_train = os.path.join(self.data_dir, self.train_path_file_target + ".pkl")
        self.path_target_test = os.path.join(self.data_dir, self.test_path_file_target + ".pkl")

        if not os.path.exists(self.path_target_train) or not os.path.exists(self.path_target_test):
            # Create vocabularies of the appropriate sizes.
            self.create_vocabulary(self.train_path_file)

            # Create token ids for the training data.
            input_train_path = self.train_path_file
            target_train_path = self.train_path_file_target
            train_input, train_input_length, train_labels = self.data_to_token_ids(input_train_path, target_train_path)

            # Create token ids for the validation data.
            input_test_path = self.test_path_file
            target_test_path = self.test_path_file_target
            test_input, test_input_length, _ = self.data_to_token_ids(input_test_path, target_test_path, train=False)

            # Collect data into a list
            training_data = [train_input, train_input_length, train_labels]
            test_data = [test_input, test_input_length]

            # Save  all the data
            with open(self.path_target_train, 'wb') as f:
                pickle.dump(training_data,f)
            with open(self.path_target_test, 'wb') as f:
                pickle.dump(test_data, f)
        else:
            # Load data
            with open(self.path_target_train, 'rb') as f:
                training_data = pickle.load(f)
            with open(self.path_target_test, 'rb') as f:
                test_data = pickle.load(f)

            # Initialize vocabulary
            self.initialize_vocabulary()

        # Convert list into a numpy array - train data
        train_input = pd.DataFrame(training_data[0]).fillna(value=0).astype(int).values
        train_length_input = np.array(training_data[1], dtype=int)
        train_labels = np.array(training_data[2], dtype=int)

        # Convert list into a numpy array - test data
        test_input = pd.DataFrame(test_data[0]).fillna(value=0).astype(int).values
        test_length_input = pd.DataFrame(test_data[1]).fillna(value=0).astype(int).values

        # Printing maximum length
        print("Shape of the input training matrix {}".format(str(train_input.shape)))
        print("Shape of the input test matrix {}".format(str(test_input.shape)))

        # Copy the files
        self.copy_files()

        # Return output
        return train_input, train_length_input, train_labels, test_input, test_length_input

    def delete_files(self):
        # Delete files
        os.remove(self.path_target_train)
        os.remove(self.path_target_test)
        os.remove(self.path_vocab)
        os.remove(self.path_tweets)

    def copy_files(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        copy2(self.path_target_train, self.output_dir)
        copy2(self.path_target_test, self.output_dir)
        copy2(self.path_vocab, self.output_dir)
        copy2(self.path_tweets, self.output_dir)
