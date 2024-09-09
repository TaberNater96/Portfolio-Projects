from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic_setup import (
    create_umap, 
    create_hdbscan, 
    create_vectorizer, 
    create_embedding_model,
    create_keybert,
    create_mmr
)
from topic_model.tm_config import BERTopicConfig, SaveConfig
from typing import List, Tuple

class CLIMATopic:
    """
    A class for topic modeling using BERTopic with custom configurations. This topic
    model is built off of the BERTopic model, but fine tuned for climate change vocabulary.
    Configurations are predefined and specified elsewhere, allowing for easy hyperparameter tuning.

    Attributes:
        bertopic_config (BERTopicConfig): Configuration for BERTopic.
        save_config (SaveConfig): Configuration for saving the model.
        umap_model: UMAP model for dimensionality reduction.
        hdbscan_model: HDBSCAN model for clustering.
        vectorizer_model: Vectorizer model for text vectorization.
        embedding_model: Sentence transformer model for text embedding.
        keybert_model: KeyBERT model for keyword extraction.
        mmr_model: MMR model for diversity-based keyword extraction.
        topic_model (BERTopic): The main BERTopic model.
    """

    def __init__(self, bertopic_config: BERTopicConfig, save_config: SaveConfig):
        """
        Initialize the CLIMATopic instance with given configurations.

        This method sets up a BERTopic model with a customized architecture for advanced topic modeling.
        The BERTopic model combines several key components to perform topic extraction:

        1. Embedding Model: Converts documents into dense vector representations.
        2. Dimensionality Reduction (UMAP): Reduces the high-dimensional embeddings to a manageable size.
        3. Clustering (HDBSCAN): Groups similar documents together in the reduced space.
        4. Vectorizer: Converts documents into sparse vector representations for topic word extraction.
        5. Keyword Extraction: Uses KeyBERT and MMR for identifying important words in each topic.

        The process works as follows:
        - Documents are embedded using the sentence transformer.
        - UMAP reduces the dimensionality of these embeddings.
        - HDBSCAN clusters the reduced embeddings to identify topics.
        - The vectorizer is used to extract important words for each cluster.
        - KeyBERT and MMR refine the topic representations by selecting diverse, relevant keywords.

        Args:
            bertopic_config (BERTopicConfig): Configuration for BERTopic, including model parameters.
            save_config (SaveConfig): Configuration for saving the model.

        The BERTopic model is highly customizable, allowing for fine-tuning of each component:
        - top_n_words: Number of words to extract per topic.
        - verbose: Controls the verbosity of the model during fitting.
        - nr_topics: Can limit the number of topics (if None, it's determined automatically).
        - seed_topic_list: Allows for guided topic modeling with predefined seed topics.
        """
        self.bertopic_config = bertopic_config
        self.save_config = save_config
        
        # Initialize all component models
        self.umap_model = create_umap()
        self.hdbscan_model = create_hdbscan()
        self.vectorizer_model = create_vectorizer()
        self.embedding_model = create_embedding_model()
        self.keybert_model = create_keybert()
        self.mmr_model = create_mmr()
        
        # Initialize the BERTopic model with all components and configurations
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,  # creating document embeddings
            umap_model=self.umap_model,            # dimensionality reduction
            hdbscan_model=self.hdbscan_model,      # clustering
            vectorizer_model=self.vectorizer_model,  # creating document-term matrix
            top_n_words=self.bertopic_config.top_n_words,  # number of words per topic
            verbose=self.bertopic_config.verbose,
            nr_topics=self.bertopic_config.nr_topics,  # limit number of topics
            seed_topic_list=self.bertopic_config.seed_topics,  # guided topic modeling
            representation_model={
                "KeyBERT": self.keybert_model,  # keyword extraction
                "MMR": self.mmr_model           # diverse keyword selection
            }
        )

    def fit(self, docs: List[str]) -> Tuple[List[int], List[float]]:
        """
        Fit the topic model to the given documents.

        Args:
            docs (List[str]): List of document strings to be modeled.

        Returns:
            Tuple[List[int], List[float]]: A tuple containing:
                - List of topic assignments for each document.
                - List of probability scores for the assigned topics.
        """
        topics, probs = self.topic_model.fit_transform(docs)
        return topics, probs

    def save(self) -> None:
        """
        Save the trained topic model.
        """
        self.topic_model.save(
            self.save_config.path,
            serialization=self.save_config.serialization,
            save_ctfidf=self.save_config.save_ctfidf,
            save_embedding_model=self.save_config.save_embedding_model
        )

    def load(self, path: str) -> None:
        """
        Load a previously saved topic model.

        Args:
            path (str): The path to the saved model file.
        """
        self.embedding_model = SentenceTransformer(self.bertopic_config.embedding_model_name)
        self.topic_model = BERTopic.load(path, embedding_model=self.embedding_model)
        
    def to_device(self, device):
        self.embedding_model = self.embedding_model.to(device)
        self.topic_model.embedding_model = self.embedding_model
        
if __name__ == "__main__":
    from topic_model.tm_config import BERTopicConfig, SaveConfig

    # Sample data
    docs = [
        "Climate change is affecting global temperatures.",
        "Renewable energy is crucial for sustainability.",
        "Carbon emissions are a major concern for the environment.",
        "Sea levels are rising due to global warming.",
        "Sustainable practices can help mitigate climate change."
    ]

    bertopic_config = BERTopicConfig()
    save_config = SaveConfig()

    climate_topic = CLIMATopic(bertopic_config, save_config)

    topics, probs = climate_topic.fit(docs)

    print("Topics assigned to documents:")
    for doc, topic in zip(docs, topics):
        print(f"Document: {doc[:30]}... | Topic: {topic}")

    climate_topic.save()
    print("\nModel saved successfully.")

    climate_topic.load(save_config.path)
    print("Model loaded successfully.")

    climate_topic.to_device('cuda')
    print("Model moved to GPU.")