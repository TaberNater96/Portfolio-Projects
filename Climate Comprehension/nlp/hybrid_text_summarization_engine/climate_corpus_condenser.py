import numpy as np
import random
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
import transformers
from transformers import (
    AutoTokenizer, 
    TFAutoModelForSeq2SeqLM, 
    AdamWeightDecay
)

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy
tf.random.set_seed(10) # tensorflow
transformers.set_seed(10) # transformers
torch.manual_seed(10) # torch
if torch.cuda.is_available(): # GPU
    torch.cuda.manual_seed_all(10)

class ClimateExtractiveModel: 
    """
    A model for extractive text summarization.

    This model selects important sentences from the original text to form a summary.
    It ranks sentences based on various features such as TF-IDF scores, cosine similarity
    of sentence embeddings, presence of named entities, keywords, and topics.

    Attributes:
        top_n (int): Number of top sentences to extract for the summary.
    """
    def __init__(
        self, 
        top_n: int
    ) -> None:
        self.top_n = top_n

    def rank_sentences(
        self, 
        sentences: List[str], 
        features: Dict[str, Any]
    ) -> List[Tuple[float, str]]:
        """
        Rank sentences based on their combined scores of various features.

        Args:
            sentences (List[str]): List of sentences to rank.
            features (Dict[str, Any]): Dictionary containing features like TF-IDF scores, 
                sentence embeddings, named entities, keywords, and LDA topics.

        Returns:
            List[Tuple[float, str]]: List of tuples containing scores and sentences, 
                sorted by score in descending order.

        Raises:
            KeyError: If a required feature is missing from the features dictionary.
            ValueError: If the number of sentences doesn't match the feature data.
        """
        try:
            tfidf_scores = features['tfidf']
            sentence_embeddings = features['embeddings']
            named_entities = features['entities']
            keywords = features['keywords']
            topics = features['topics']
        except KeyError as e:
            raise KeyError(f"Missing required feature: {str(e)}")

        if len(sentences) != len(sentence_embeddings) or len(sentences) != len(tfidf_scores):
            raise ValueError("Number of sentences doesn't match the embedding or TF-IDF data.")

        # Calculate cosine similarity matrix for sentence embeddings
        cosine_sim_matrix = cosine_similarity(sentence_embeddings)

        # Combine TF-IDF scores and cosine similarity of embeddings
        combined_scores = np.zeros(len(sentences))
        for i, sentence in enumerate(sentences):
            combined_scores[i] += tfidf_scores[i]
            
            # Use sum of cosine similarities as a proxy for sentence importance
            combined_scores[i] += cosine_sim_matrix[i].sum()
            
            # Boost sentences containing named entities, keywords, and top topics
            if any(entity in sentence for entity, _ in named_entities):
                combined_scores[i] += 1.0  # boost for NER
            if any(keyword in sentence for keyword in keywords):
                combined_scores[i] += 0.5  # boost for keywords
            if any(topic in sentence for topic in topics['topics'].keys()):
                combined_scores[i] += 0.5  # boost for top topics

        # Sort the sentences based on their combined scores in descending order
        ranked_sentences = sorted(
            # Each sentence is paired with its score (as a tuple), and the list of tuples is sorted by score
            ((score, sent) for score, sent in zip(combined_scores, sentences)),
            key=lambda x: x[0], # sort by score in descending order
            reverse=True
        )

        return ranked_sentences

    def summarize(
        self, 
        sentences: List[str], 
        features: Dict[str, List[Any]], 
        article_index: int
    ) -> List[str]:
        """
        Generate a summary by extracting the top N ranked sentences.

        Args:
            sentences (List[str]): List of sentences to summarize.
            features (Dict[str, List[Any]]): Dictionary containing features for all articles.
            article_index (int): Index of the article to summarize.

        Returns:
            List[str]: List of top N ranked sentences forming the summary.

        Raises:
            ValueError: If the number of sentences is less than top_n.
        """
        if len(sentences) < self.top_n:
            raise ValueError(f"Number of sentences ({len(sentences)}) is less than top_n ({self.top_n})")

        # Extract features for this specific article
        article_features = {
            key: value[article_index] for key, value in features.items()
        }

        # Rank the sentences based on their features and extract the top N sentences
        ranked_sentences = self.rank_sentences(sentences, article_features)
        top_sentences = [sent for _, sent in ranked_sentences[:self.top_n]]
        return top_sentences

class CLIMATEBart:
    """
    A model for abstractive text summarization using a pre-trained BART model.

    This model generates a summary by understanding the content and creating new sentences,
    rather than extracting existing sentences from the text.

    Attributes:
        tokenizer: The tokenizer used to process input text.
        model: The pre-trained BART model for sequence-to-sequence tasks.
        optimizer: The optimizer used for training the model.
    """
    def __init__(
        self, 
        model_name: str = "facebook/bart-large-cnn", 
        learning_rate: float = 2e-5
    ) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.optimizer = AdamWeightDecay(learning_rate=learning_rate)
        except ImportError as e:
            raise ImportError(f"Required library not installed: {str(e)}")
        except OSError as e:
            raise OSError(f"Error loading model or tokenizer: {str(e)}")

    def train(
        self, 
        train_texts: List[str], 
        val_texts: List[str], 
        train_summaries: List[str], 
        val_summaries: List[str], 
        batch_size: int = 32,
        epochs: int = 6
    ) -> None:
        """
        Train the abstractive model using provided training and validation data.

        Args:
            train_texts (List[str]): List of training texts.
            val_texts (List[str]): List of validation texts.
            train_summaries (List[str]): List of training summaries.
            val_summaries (List[str]): List of validation summaries.
            batch_size (int): Batch size for training. Defaults to 32.
            epochs (int): Number of epochs for training. Defaults to 3.

        Raises:
            ValueError: If input data lengths don't match or are empty.
        """
        if not (len(train_texts) == len(train_summaries) and len(val_texts) == len(val_summaries)):
            raise ValueError("Mismatch in the number of texts and summaries")
        if not (train_texts and val_texts and train_summaries and val_summaries):
            raise ValueError("Empty input data")

        train_encodings = self.tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
        
        val_encodings = self.tokenizer(
            val_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_summaries
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_summaries
        ))

        # Batch the datasets for training and validation
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        self.model.compile(optimizer=self.optimizer)
        self.model.fit(
            train_dataset, 
            validation_data=val_dataset, 
            epochs=epochs
        )

    def save_model(
        self, 
        path: str
    ) -> None:
        """
        Save the model and tokenizer to the specified path.

        Args:
            path (str): Directory path to save the model and tokenizer.

        Raises:
            OSError: If there's an error saving the model or tokenizer.
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except OSError as e:
            raise OSError(f"Error saving model or tokenizer: {str(e)}")

    def load_model(
        self, 
        path: str
    ) -> None:
        """
        Load the model and tokenizer from the specified path.

        Args:
            path (str): Directory path to load the model and tokenizer from.

        Raises:
            OSError: If there's an error loading the model or tokenizer.
        """
        try:
            self.model = TFAutoModelForSeq2SeqLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except OSError as e:
            raise OSError(f"Error loading model or tokenizer: {str(e)}")

    def generate_summary(
        self, 
        text: str, 
        max_length: int = 250, 
        min_length: int = 50, 
        length_penalty: float = 2.0, 
        num_beams: int = 4, 
        early_stopping: bool = True
    ) -> str:
        """
        Generate a summary for the given text.

        Args:
            text (str): The input text to summarize.
            max_length (int): Maximum length of the generated summary. Defaults to 150.
            min_length (int): Minimum length of the generated summary. Defaults to 40.
            length_penalty (float): Exponential penalty to the length. Defaults to 2.0.
            num_beams (int): Number of beams for beam search. Defaults to 4.
            early_stopping (bool): Whether to stop the beam search when at least num_beams 
                sentences are finished per batch or not. Defaults to True.

        Returns:
            str: The generated summary.

        Raises:
            ValueError: If the input text is empty.
        """
        if not text.strip():
            raise ValueError("Input text is empty")

        inputs = self.tokenizer.encode(
            "summarize: " + text, 
            return_tensors="tf", 
            truncation=True, 
            padding=True
        )
        
        outputs = self.model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=length_penalty, 
            num_beams=num_beams, 
            early_stopping=early_stopping
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class ClimateCorpusCondenser:
    """
    A hybrid model combining extractive and abstractive summarization techniques.

    This model first applies extractive summarization to select important sentences,
    then uses abstractive summarization to refine and generate the final summary.

    Attributes:
        extractive_model (ClimateExtractiveModel): The extractive summarization model.
        abstractive_model (AbstractiveModel): The abstractive summarization model.
    """
    def __init__(
        self, 
        abstractive_model: CLIMATEBart
    ) -> None:
        self.abstractive_model = abstractive_model

    def train(
        self, 
        articles: List[str], 
        summaries: List[str],
        extractive_model: ClimateExtractiveModel,
        features: Dict[str, List[Any]],
        test_size: float = 0.1,
        random_state: int = 10
    ) -> None:
        """
        Train the hybrid summarization model using both extractive and abstractive techniques.

        This method first splits the data into training and validation sets, then applies
        extractive summarization to both sets. Finally, it trains the abstractive model
        using the extractive summaries and the original summaries, resulting in a hybrid model
        that combines the power of both extractive and abstractive techniques.

        Args:
            articles (List[str]): List of input articles for training.
            summaries (List[str]): List of corresponding summaries for the input articles.
            extractive_model (ClimateExtractiveModel): An instance of the extractive summarization model.
            features (Dict[str, List[Any]]): Pre-computed features for all articles.
            test_size (float, optional): Proportion of the dataset to include in the validation split. 
                                        Defaults to 0.1.
            random_state (int, optional): Controls the shuffling applied to the data before applying the split. 
                                        Defaults to 10.

        Raises:
            ValueError: If the number of articles doesn't match the number of summaries.
        """
        if len(articles) != len(summaries):
            raise ValueError("Number of articles doesn't match the number of summaries")

        train_articles, val_articles, train_summaries, val_summaries = train_test_split(
            articles, 
            summaries, 
            test_size=test_size, 
            random_state=random_state
        )

        train_ext_summaries = [extractive_model.summarize(article.split(), features, i) 
                               for i, article in enumerate(train_articles)]
        val_ext_summaries = [extractive_model.summarize(article.split(), features, i) 
                             for i, article in enumerate(val_articles)]

        self.abstractive_model.train(
            train_ext_summaries, 
            val_ext_summaries, 
            train_summaries, 
            val_summaries
        )

    def summarize(
        self, 
        article: str
    ) -> str:
        """
        Generate a final summary by first applying the extractive model, then refining with the abstractive model.

        Args:
            article (str): The input article to summarize.

        Returns:
            str: The final summary.

        Raises:
            ValueError: If the input article is empty.
        """
        if not article.strip():
            raise ValueError("Input article is empty")

        return self.abstractive_model.generate_summary(article)
    
    def save(
        self, 
        path: str
    ) -> None:
        """
        Save the hybrid model to the specified path.

        Args:
            path (str): Directory path to save the model.
        """
        self.abstractive_model.save_model(path)

    def load(
        self, 
        path: str
    ) -> None:
        """
        Load the hybrid model from the specified path.

        Args:
            path (str): Directory path to load the model from.
        """
        self.abstractive_model.load_model(path)
        
if __name__ == "__main__":
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Sample data
    articles = [
        "Climate change is a pressing global issue. Rising temperatures are causing extreme weather events. Governments worldwide are implementing policies to reduce carbon emissions.",
        "Renewable energy sources are becoming increasingly important. Solar and wind power are growing in popularity. Many countries are investing in green technology to combat climate change."
    ]
    summaries = [
        "Climate change causes extreme weather and prompts government action.",
        "Renewable energy growth helps combat climate change."
    ]

    # Initialize models
    abstractive_model = CLIMATEBart()
    extractive_model = ClimateExtractiveModel(top_n=2) # top 2 sentences
    C3 = ClimateCorpusCondenser(abstractive_model)

    # Prepare features (simplified for this example)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(articles)
    features = {
        'tfidf': tfidf_matrix.toarray().tolist(),
        'embeddings': np.random.rand(len(articles), 300).tolist(),
        'entities': [[] for _ in articles],
        'keywords': [[] for _ in articles], 
        'topics': [{'topics': {}} for _ in articles] 
    }

    # Train the model
    C3.train(articles, summaries, extractive_model, features)

    # Generate a summary for a new article
    new_article = "Global temperatures have risen significantly over the past century. This has led to melting ice caps, rising sea levels, and more frequent natural disasters. Scientists warn that urgent action is needed to mitigate the effects of climate change."
    summary = C3.summarize(new_article)

    print("Original article:")
    print(new_article)
    print("\nGenerated summary:")
    print(summary)

    # Save the model
    C3.save("path/C3")

    # Load the model (uncomment to test loading)
    # climate_condenser.load("path/C3")