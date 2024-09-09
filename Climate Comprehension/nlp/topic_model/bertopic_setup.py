from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from topic_model.tm_config import (
    BERTopicConfig, 
    HDBSCANConfig, 
    UMAPConfig, 
    CountVectorizerConfig, 
    KeyBERTConfig, 
    MMRConfig
)

def create_umap() -> UMAP:
    """
    Creates a UMAP (Uniform Manifold Approximation and Projection) model for dimensionality reduction.

    UMAP is a technique that projects high-dimensional data into a lower-dimensional space
    while preserving both local and global structure. This function configures UMAP to:
    1. Reduce the dimensionality of text embeddings, typically to 2D or 3D.
    2. Preserve the relationships between similar documents in the reduced space.
    3. Enable efficient clustering and visualization of the document space.

    The function uses predefined settings from UMAPConfig, which can be adjusted to 
    balance between preserving local versus global structure in the data.

    Returns:
        UMAP: A configured UMAP model ready for fitting and transforming data.
    """
    umap_config = UMAPConfig()
    return UMAP(
        n_neighbors=umap_config.n_neighbors, # number of neighbors, higher neighbors = more concentrated
        n_components=umap_config.n_components, # number of dimensions to reduce to
        min_dist=umap_config.min_dist, # how close points should be to each other
        metric=umap_config.metric, # distance metric
        random_state=umap_config.random_state # random state for reproducibility
    )

def create_hdbscan() -> HDBSCAN:
    """
    Creates an HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) model for clustering.

    HDBSCAN is an unsupervised learning algorithm that:
    1. Identifies clusters of varying densities in the data.
    2. Handles noise points that don't belong to any cluster.
    3. Determines the number of clusters automatically based on the data structure.

    This function configures HDBSCAN to work effectively with the reduced-dimensional
    embeddings produced by UMAP. It's particularly useful for topic modeling as it can:
    - Identify coherent groups of documents (potential topics)
    - Handle outlier documents that don't fit well into any topic
    - Adapt to the natural number of topics in the dataset

    The settings from HDBSCANConfig control the granularity and size of detected clusters.

    Returns:
        HDBSCAN: A configured HDBSCAN model ready for fitting to data and predicting clusters.
    """
    hdbscan_config = HDBSCANConfig()
    return HDBSCAN(
        min_cluster_size=hdbscan_config.min_cluster_size, # number of documents in a cluster
        min_samples=hdbscan_config.min_samples, # none
        metric=hdbscan_config.metric, # distance metric
        cluster_selection_method=hdbscan_config.cluster_selection_method, # method for selecting clusters
        prediction_data=hdbscan_config.prediction_data # whether to return cluster labels
    )

def create_vectorizer() -> CountVectorizer:
    """
    Creates a CountVectorizer for converting text documents into a bag-of-words representation.

    This function sets up a CountVectorizer that:
    1. Tokenizes the input text into individual words or n-grams.
    2. Counts the occurrences of each token in each document.
    3. Creates a sparse matrix where each row represents a document and each column a unique token.

    The vectorizer is configured to:
    - Remove common stop words that typically don't contribute to the meaning.
    - Consider only tokens that appear in a minimum number of documents (controlled by min_df).
    - Generate n-grams (sequences of n adjacent words) to capture phrases.

    This vectorization is crucial for topic modeling as it creates a numerical representation
    of the documents that can be analyzed for frequent terms and co-occurrences, which form
    the basis of discovered topics.

    Returns:
        CountVectorizer: A configured CountVectorizer ready to fit on a corpus and transform documents.
    """
    vectorizer_config = CountVectorizerConfig()
    return CountVectorizer(
        stop_words=vectorizer_config.stop_words, # remove common english stop words
        min_df=vectorizer_config.min_df, # ignore terms that appear in less than 2 documents
        ngram_range=vectorizer_config.ngram_range 
    )

def create_embedding_model() -> SentenceTransformer:
    """
    Creates a SentenceTransformer model for generating dense vector representations of text.

    This function loads a pre-trained SentenceTransformer model that:
    1. Takes text input (sentences or documents) and produces fixed-size dense vector embeddings.
    2. Captures semantic meaning, allowing similar texts to have similar embeddings.
    3. Provides a rich, contextual representation of each document.

    The specific model used is defined in BERTopicConfig. These embeddings are crucial for topic modeling as they:
    - Allow for semantic similarity comparisons between documents.
    - Provide input for dimensionality reduction (UMAP) and clustering (HDBSCAN).
    - Enable more nuanced topic representations compared to simple bag-of-words approaches.

    Returns:
        SentenceTransformer: A loaded SentenceTransformer model ready to encode text into embeddings.
    """
    bertopic_config = BERTopicConfig()
    return SentenceTransformer(bertopic_config.embedding_model_name)

def create_keybert() -> KeyBERTInspired:
    """
    Creates a KeyBERTInspired model for extracting keywords from topics.

    This function configures a KeyBERTInspired object that:
    1. Identifies the most representative words or phrases for each topic.
    2. Uses the underlying embedding model to find words that are semantically central to the topic.
    3. Provides a method to summarize topics with key terms.

    The model is set up to:
    - Extract a specified number of top words (controlled by top_n_words).
    - Use a consistent random seed for reproducibility.

    KeyBERTInspired improves upon simple frequency-based keyword extraction by considering
    the semantic relationships between words, leading to more meaningful and coherent topic representations.

    Returns:
        KeyBERTInspired: A configured KeyBERTInspired model ready to extract keywords from topics.
    """
    keybert_config = KeyBERTConfig()
    return KeyBERTInspired(
        top_n_words=keybert_config.top_n_words,
        random_state=keybert_config.random_state
    )

def create_mmr() -> MaximalMarginalRelevance:
    """
    Creates a MaximalMarginalRelevance (MMR) model for diverse keyword selection within topics.

    This function sets up an MMR model that:
    1. Selects a diverse set of keywords to represent each topic.
    2. Balances between the relevance of keywords and their diversity within the set.
    3. Helps prevent redundancy in topic representations.

    The diversity parameter controls the trade-off between relevance and diversity:
    - Higher values prioritize diversity, ensuring a broader representation of the topic.
    - Lower values prioritize relevance, potentially at the cost of some redundancy.

    MMR is particularly useful in topic modeling as it:
    - Provides a more comprehensive view of each topic by including varied aspects.
    - Helps distinguish between similar topics by highlighting their unique elements.
    - Improves the interpretability of topics by avoiding repetitive keywords.

    Returns:
        MaximalMarginalRelevance: A configured MMR model ready to select diverse keywords.
    """
    mmr_config = MMRConfig()
    return MaximalMarginalRelevance(diversity=mmr_config.diversity)

if __name__ == "__main__":
    print("Creating UMAP model...")
    umap_model = create_umap()
    print("UMAP model created.")

    print("\nCreating HDBSCAN model...")
    hdbscan_model = create_hdbscan()
    print("HDBSCAN model created.")

    print("\nCreating vectorizer...")
    vectorizer = create_vectorizer()
    print("Vectorizer created.")

    print("\nCreating embedding model...")
    embedding_model = create_embedding_model()
    print("Embedding model created.")

    print("\nCreating KeyBERT model...")
    keybert_model = create_keybert()
    print("KeyBERT model created.")

    print("\nCreating MMR model...")
    mmr_model = create_mmr()
    print("MMR model created.")

    print("\nAll features for BERTopic have been successfully created.")