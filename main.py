# CIL Project
# Main file
# Description: The script loads a model and performs training or predictions
# Import site-packages libraries

# Runs the main script and all the dependencies
from utils import Preprocessing
from configuration import get_configuration
from configuration import print_configuration
from embeddings.embedding_glove import GloVeModel
from embeddings.embedding_word2vec_gem import Word2VecModel

# Select model to run
from models.rnn_model import RNN_Model

def main():
    # Setup and get current configuration
    config = get_configuration()
    # Print parameters
    print_configuration()
    #Initialize class - preprocessing
    preprocess = Preprocessing(config = config)
    # Perform preprocessing
    train_input, train_length_input, train_labels, test_input, test_length_input = preprocess.prepare_data()
    # Initialize class and select mode and model - embeddings
    if config.mode != "infer":
        if config.emb_model == "glove":
            model_emb = GloVeModel(config=config, dict_vocab=preprocess.dict_vocab)
        else:
            model_emb = Word2VecModel(config=config, dict_vocab=preprocess.dict_vocab)
        # Fit corpus
        model_emb.fit_to_corpus()
        # Train embeddings
        model_emb.train()
    # Train model
    RNN_Model(config, preprocess.dict_vocab_reverse, train_input, train_length_input,
              train_labels, test_input, test_length_input)
if __name__ == '__main__':
    main()
