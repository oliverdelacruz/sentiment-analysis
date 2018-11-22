########################################################################################################################
#WORD2VEC MODEL
########################################################################################################################
from gensim.models import Phrases, Word2Vec, KeyedVectors
# Import libaries
import os
import time
import datetime
import pickle
import numpy as np


# Import config file
class Word2VecModel():
    def __init__(self, dict_vocab = None,  config = None):

        # Parameters defined in configuration file
        self.config = config
        self.out_dir = config.output_dir
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size + config.vocab_tags
        self.data_dir = config.data_dir
        self.vocab_path_file = config.vocab_file
        self.sentence_path_file = config.tweets_file
        self.dict_vocab = dict_vocab
        self.window_size = config.window_size
        self.emb_dir =  config.emb_dir
        self.iter = config.iter

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

    def fit_to_corpus(self):
        # Load tweets
        path_tweets = os.path.join(self.data_dir, self.sentence_path_file)
        with open(path_tweets, 'rb') as f:
            corpus = pickle.load(f)

        # Conversion from integers to words
        self.corpus = self.add_tags(corpus)

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

    def train(self):
        # Save the model
        self.out_dir_file = os.path.join(self.emb_dir, "word2vec_emb_dim-{}_window_size-{}_vocab_size-{}_iter-{}.npy".
                                         format(self.embedding_dim, self.window_size, self.vocab_size, self.iter))
        print(self.out_dir_file)
        # Train the model
        if (not os.path.isfile(self.out_dir_file)):
            print("Training embeddings from %s" % self.emb_dir)
            model = Word2Vec(self.corpus, size=self.embedding_dim, window=self.window_size, min_count=5, workers=4, iter=self.iter)

            # Initialize embeddings
            external_embedding = np.zeros(shape=(self.vocab_size, self.embedding_dim))

            # Recover matrix
            matches = 0
            for tok, idx in self.word_to_id.items():
                if tok in model.wv.vocab:
                    external_embedding[idx] = model.wv[tok]
                    matches += 1
                else:
                    print("%s not in embedding file" % tok)
                    external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=self.embedding_dim)

            # Save files
            np.save(self.out_dir_file, external_embedding)
            print("%d words out of %d could be loaded" % (matches, self.vocab_size))

        # Load file and save it
        external_embedding = np.load(self.out_dir_file)
        np.save(os.path.join(self.emb_dir, "embeddings_word2vec"), external_embedding)
        del external_embedding


    def train_emb_google(self):
        # Save files
        self.out_dir_file = os.path.join(self.emb_dir,
                                         "word2vec_emb_dim-{}_window_size-{}_vocab_size-{}_iter-{}_google.npy".
                                         format(self.embedding_dim, self.window_size, self.vocab_size, self.iter))

        # Load model
        if (not os.path.isfile(self.out_dir_file)):
            print("Loading external embeddings from %s" % self.emb_dir)
            external_embedding = np.zeros(shape=(self.vocab_size, 300))
            self.out_dir_file = os.path.join(self.emb_dir, 'GoogleNews-vectors-negative300.bin')
            model = KeyedVectors.load_word2vec_format(self.out_dir_file, binary=True)
            matches = 0
            for tok, idx in self.word_to_id.items():
                if tok in model.vocab:
                    external_embedding[idx] = model[tok]
                    matches += 1
                else:
                    print("%s not in embedding file" % tok)
                    external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=300)
            print("%d words out of %d could be loaded" % (matches, self.vocab_size))

        # Save files
        external_embedding = np.load(self.out_dir_file)
        np.save(os.path.join(self.emb_dir, "embeddings_word2vec"), external_embedding)
        del external_embedding

    def train_extra(self):
        # Save the model
        self.out_dir_file = os.path.join(self.emb_dir,
                                         "word2vec_emb_dim-{}_window_size-{}_vocab_size-{}_iter-{}_google_instersect.npy".
                                         format(self.embedding_dim, self.window_size, self.vocab_size, self.iter))

        # Load model
        if (not os.path.isfile(self.out_dir_file)):
            print("Loading external embeddings from %s" % self.emb_dir)
            external_embedding = np.zeros(shape=(self.vocab_size, 300))
            self.out_dir_file = os.path.join(self.emb_dir, 'GoogleNews-vectors-negative300.bin')
            model = Word2Vec(self.corpus, size=300, window=self.window_size, min_count=5, workers=4, iter=self.iter)
            model.intersect_word2vec_format(self.out_dir_file, lockf=1.0, binary=True)
            model.train(self.corpus, total_examples=model.corpus_count,  epochs=model.iter)
            matches = 0
            for tok, idx in self.word_to_id.items():
                if tok in model.wv.vocab:
                    external_embedding[idx] = model.wv[tok]
                    matches += 1
                else:
                    print("%s not in embedding file" % tok)
                    external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=300)
            print("%d words out of %d could be loaded" % (matches, self.vocab_size))

        # Load file and save it
        external_embedding = np.load(self.out_dir_file)
        np.save(os.path.join(self.emb_dir, "embeddings_word2vec"), external_embedding)
        del external_embedding









