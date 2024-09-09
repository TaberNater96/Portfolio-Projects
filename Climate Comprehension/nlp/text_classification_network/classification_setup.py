"""
------------------------------------------------------------------------------------------------------------
                                Advanced NLP Preprocessor
------------------------------------------------------------------------------------------------------------

By: Elijah Taber
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

class LexicalTokenizer:
    """
    A class designed to handle the tokenization of a dataframe containing normalized text,
    topics, classifications, sentiment, and summaries using the Keras tokenizer. The tokenizer is 
    trained on the corpus column of the training dataframe, and creates a word-index and index-word
    vocabulary, that represent the mapping between words and integer indices. This tokenizer is applied
    to the entire corpus column of the dataframe, not just individual rows. Hence a large amount of text
    will be used to train, resulting in a need to set a maximum word limit so that the tokenizer can 
    actully learn what words are important based on term frequency.

    Methods for fitting and tokenizing the dataframe:
    -   fit()
    -   tokenize_text()
    -   pad_text()
    -   encode_labels()

    Methods to access the word-index and index-word dictionaries:
    -  word_index()
    -  index_word()

    Methods to save and load the tokenizer:
    -   save_tokenizer()
    -   load_tokenizer()

    Attributes:
        tokenizer (Tokenizer): Keras tokenizer instance used for tokenizing text.
        max_seq_length (int): Maximum length of sequences after padding.
        embed (Module): Universal Sentence Encoder module for generating sentence embeddings.
    """

    def __init__(
        self, 
        num_words: int = 10_000, 
        max_seq_length: int = 100, 
        use_model_path: str = 'use_model'
    ):
        """
        Initializes the LexicalTokenizer instance with a Keras tokenizer and configuration parameters.
        The num_words parameter sets a limit on the maximum number of words to keep in the tokenizer’s 
        vocabulary, based on their frequency. The max_seq_length parameter determines the maximum length 
        of sequences after padding. Sequences longer than this length will be truncated, while shorter 
        ones will be padded. This ensures uniform input size for the neural network, which is crucial 
        for batch processing in deep learning models. The universal sentence encoder is defined as a secondary
        use case for a sentence level context representation. This allows the class to be more versital and not
        rely solely on word level embeddings.

        Parameters:
            num_words (int): Maximum number of words to keep, based on word frequency. Only the most common num_words
                             words will be kept. Default is 10_000.
            max_seq_length (int): Maximum length of sequences after padding. Sequences longer than this will be truncated,
                                  and shorter ones will be padded. Default is 100, this is the same dimensions as the GloVe 
                                  embedding.
            use_model_path (str): Path to the saved Universal Sentence Encoder model.
        """
        # Initialize the Keras Tokenizer with a maximum vocabulary size of 25,000 words and an out-of-vocabulary token for
        # words that are not in the vocabulary
        self.tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
        self.max_seq_length = max_seq_length
        self.label_encoders = {
            'topic': LabelEncoder(),
            'classification': LabelEncoder(),
            'sentiment': LabelEncoder()
        }
        self.embed = hub.load(use_model_path)

    def fit(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        topic_column: str, 
        classification_column: str, 
        sentiment_column: str
    ) -> None:
        """
        Fits the tokenizer on the text column and the label encoders on the categorical columns.

        Parameters:
            df (pd.DataFrame): The dataframe containing the columns to be processed.
            text_column (str): The name of the column containing text to be tokenized.
            topic_column (str): The name of the column containing topic labels to be encoded.
            classification_column (str): The name of the column containing classification labels to be encoded.
            sentiment_column (str): The name of the column containing sentiment labels to be encoded.
        """
        self.tokenizer.fit_on_texts(df[text_column])
        self.label_encoders['topic'].fit(df[topic_column])
        self.label_encoders['classification'].fit(df[classification_column])
        self.label_encoders['sentiment'].fit(df[sentiment_column])

    def tokenize_text(self, text: str) -> List[int]:
        """
        Converts a single piece of text into a sequence of integers using the vocabulary built by the tokenizer.

        Parameters:
            text (str): The text to be tokenized.

        Returns:
            List[int]: Tokenized sequence of integers.
        """
        return self.tokenizer.texts_to_sequences([text])[0]

    def pad_text(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pads a list of sequences to ensure that all sequences have the same length, specified by the max_seq_length parameter.

        Parameters:
            sequences (List[List[int]]): List of sequences to be padded.

        Returns:
            np.ndarray: Padded sequences as a numpy array.
        """
        return pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')

    def encode_labels(self, df: pd.DataFrame, column: str) -> np.ndarray:
        """
        Encodes the labels in the specified column using the label encoder.

        Parameters:
            df (pd.DataFrame): The dataframe containing the column to be encoded.
            column (str): The name of the column to be encoded.

        Returns:
            np.ndarray: Encoded labels.
        """
        return self.label_encoders[column].transform(df[column])

    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str, 
        topic_column: str, 
        classification_column: str, 
        sentiment_column: str, 
        summary_column: str
    ) -> Dict[str, np.ndarray]:
        """
        Preprocesses the dataframe by tokenizing and padding the text and summary columns,
        and encoding the topic, classification, and sentiment columns. Additionally, generates
        embeddings using the Universal Sentence Encoder.

        Parameters:
            df (pd.DataFrame): The dataframe containing the columns to be processed.
            text_column (str): The name of the column containing text to be tokenized and padded.
            topic_column (str): The name of the column containing topic labels to be encoded.
            classification_column (str): The name of the column containing classification labels to be encoded.
            sentiment_column (str): The name of the column containing sentiment labels to be encoded.
            summary_column (str): The name of the column containing summary text to be tokenized and padded.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the processed data arrays.
        """
        # Convert tokenized text and summary to fixed-length sequences
        tokenized_text = self.tokenizer.texts_to_sequences(df[text_column])
        tokenized_summary = self.tokenizer.texts_to_sequences(df[summary_column])
        
        # Pad text and summary sequences to the same length
        padded_text = self.pad_text(tokenized_text)
        padded_summary = self.pad_text(tokenized_summary)
        
        # Generate embedding matrices for text and summary using Universal Sentence Encoder
        text_embeddings = self.embed(df[text_column].tolist()).numpy()
        summary_embeddings = self.embed(df[summary_column].tolist()).numpy()
        
        # Encode categorical labels
        encoded_topic = self.encode_labels(df, 'topic')
        encoded_classification = self.encode_labels(df, 'classification')
        encoded_sentiment = self.encode_labels(df, 'sentiment')
        
        # Return processed data
        return {
            'text': padded_text,
            'summary': padded_summary,
            'topic': encoded_topic,
            'classification': encoded_classification,
            'sentiment': encoded_sentiment,
            'text_embeddings': text_embeddings,
            'summary_embeddings': summary_embeddings
        }

    def word_index(self) -> Dict[str, int]:
        """
        Returns the word index dictionary, which maps each word in the vocabulary to its corresponding integer index.

        Returns:
            Dict[str, int]: Word index dictionary.
        """
        return self.tokenizer.word_index

    def index_word(self) -> Dict[int, str]:
        """
        Returns the reverse mapping of the word index dictionary, where each integer index is mapped back to its corresponding word.

        Returns:
            Dict[int, str]: Index word dictionary.
        """
        return {index: word for word, index in self.tokenizer.word_index.items()}

    def save_tokenizer(self, file_path: str) -> None:
        """
        Saves the tokenizer to a specified file path.

        Parameters:
            file_path (str): Path to save the tokenizer.
        """
        with open(file_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self, file_path: str) -> None:
        """
        Loads the tokenizer from a specified file path.

        Parameters:
            file_path (str): Path to load the tokenizer from.
        """
        with open(file_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            
class EmbeddingMatrix:
    """
    This class is specifically designed to construct an embedding matrix utilizing pre-trained GloVe 
    (Global Vectors for Word Representation) embeddings. This class takes as input a pre-trained tokenizer, 
    which is used to convert input text into sequences of integers. The output is an embedding matrix, which
    is a multi-dimensional array of floating point numbers. Each row in this matrix corresponds to a word 
    in the tokenizer's vocabulary, and the values in that row corresponds to the GloVe embedding for that
    word. This matrix is primarily intended to be used as the weights in an embedding layer of a neural 
    network, allowing the network to utilize pre-trained word embeddings for tasks such as text 
    classification or sentiment analysis. This specific GloVe model is trained on a corpus of 6 billion
    tokens and has 100-dimensional embeddings.

    Attributes:
        tokenizer (LexicalTokenizer): An instance of LexicalTokenizer that provides the word index.
        embedding_dim (int): The dimensionality of the embeddings. Default is 100 dimensions.
        glove_path (str): The file path to the pre-trained GloVe embeddings. Default is 'GloVe/glove.6B.100d.txt'.
        embedding_matrix (np.ndarray): The generated embedding matrix.
    """
    def __init__(self, tokenizer: LexicalTokenizer, embedding_dim: int = 100, glove_path: str = 'GloVe/glove.6B.100d.txt'):
        """
        Initializes the EmbeddingMatrix instance with the specified tokenizer, embedding dimensions, 
        and GloVe file path.

        Parameters:
            tokenizer (LexicalTokenizer): An instance of LexicalTokenizer that provides the word index.
            embedding_dim (int): The dimensionality of the embeddings. Default is 100 dimensions.
            glove_path (str): The file path to the pre-trained GloVe embeddings. Default is 'GloVe/glove.6B.100d.txt'.
        """
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.glove_path = glove_path
        self.embedding_matrix = None

    def load_glove_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Loads GloVe embeddings from the specified file path and returns a dictionary of word vectors.

        Reads the GloVe file line by line, splits each line into the word and its corresponding coefficients,
        and stores these in a dictionary where the key is the word and the value is the embedding vector.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are words and values are their GloVe embedding vectors.
        """
        embeddings_index = {}
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split() # split each line into a list of values
                word = values[0] # the first value is the actual word
                coefs = np.asarray(values[1:], dtype='float32') # the rest are the coefficients
                embeddings_index[word] = coefs
        return embeddings_index

    def create_embedding_matrix(self) -> np.ndarray:
        """
        Creates the embedding matrix using the loaded GloVe embeddings and the tokenizer's word index.

        For each word in the tokenizer's word index, if the word exists in the GloVe embeddings, its vector 
        is added to the embedding matrix at the corresponding index. If a word does not have a GloVe 
        embedding, its vector remains as zeros.

        Returns:
            np.ndarray: The generated embedding matrix where each row corresponds to a word's embedding vector.
        """
        embeddings_index = self.load_glove_embeddings()
        word_index = self.tokenizer.word_index()
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim)) # +1 for the padding token
        for word, i in word_index.items(): 
            embedding_vector = embeddings_index.get(word) # get the GloVe embedding for the word
            if embedding_vector is not None: # if the word is in the GloVe embeddings
                embedding_matrix[i] = embedding_vector # add the embedding to the matrix
        self.embedding_matrix = embedding_matrix # store the embedding matrix
        return embedding_matrix

    def get_embedding_matrix(self) -> np.ndarray:
        """
        Returns the embedding matrix. If the embedding matrix has not been created yet, it creates it first.

        Returns:
            np.ndarray: The embedding matrix.
        """
        if self.embedding_matrix is None:
            self.create_embedding_matrix()
        return self.embedding_matrix