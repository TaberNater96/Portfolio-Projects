"""
By: Elijah Taber

This module, ts_feature_engineering.py, is designed to provide comprehensive feature engineering capabilities for a hybrid text 
summarization model using both extraction and abstraction feature engineering techniques, specifically tailored for summarizing 
complex, technical documents such as climate change reports. It includes a variety of feature extraction functions that prepare 
the text data by extracting meaningful and informative features, which are crucial for both extractive and abstractive 
summarization phases. This file takes a functional programming approach to defining the features and their extraction methods,
ensuring that they are easily accessible and reusable.

Features included are:
- TF-IDF scoring to highlight important terms.
- Sentence and word embeddings to capture semantic meanings.
- Named entity recognition for identifying and categorizing key information.
- LDA topic modeling to discern and quantify the main themes of the text.
- Keyword extraction to emphasize significant phrases and concepts.
"""

import pandas as pd
import random
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sentence_transformers import SentenceTransformer
import transformers
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy/sklearn
transformers.set_seed(10) # transformers

# Load NLP models
nlp = spacy.load('en_core_web_sm')

# Initialize tools
tfidf_vectorizer = TfidfVectorizer()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2') # 384 dimensions
rake_nltk_var = Rake()

def extract_tfidf_features(
    sentences: List[str]
) -> Dict[str, float]:
    """
    This function takes a list of sentences as input and returns a dictionary of words and their corresponding TF-IDF scores.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate how important
    a word is to a corpus in a collection of corpuses. The importance increases proportionally to the number 
    of times a word appears in the corpus but is offset by the frequency of the word in the corpus.

    Parameters:
        sentences (List[str]): The list of sentences for which to calculate TF-IDF scores.

    Returns:
        Dict[str, float]: A dictionary where the keys are the words from the corpus and the values are their 
        corresponding TF-IDF scores.
    """
    # Join sentences to create a single document for TF-IDF calculation
    corpus = " ".join(sentences)
    tfidf_matrix = tfidf_vectorizer.fit_transform([corpus])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    return tfidf_scores

def extract_sentence_embeddings(
    sentences: List[str]
) -> np.ndarray:
    """
    This function takes a list of sentences as input and returns sentence-level embeddings.
    
    Sentence embeddings are vector representations of sentences. These embeddings are generated using
    a pre-trained model (in this case, 'all-MiniLM-L6-v2' from Hugging Face) that has learned to map 
    sentences to a high-dimensional space where sentences with similar meanings are located close to 
    each other.

    Parameters:
        sentences (List[str]): The list of sentences for which to generate embeddings.

    Returns:
        np.ndarray: A numpy array containing the sentence embeddings for the input sentences.
    """
    embeddings = sentence_model.encode(sentences)
    return embeddings

def extract_named_entities(
    corpus: str
) -> List[Tuple[str, str]]:
    """
    This function takes a corpus as input and returns a list of named entities and their corresponding types.
    
    Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify 
    named entities in text into pre-defined categories such as person names, organizations, locations, medical
    codes, time expressions, quantities, monetary values, percentages, etc.

    Parameters:
        corpus (str): The corpus from which to extract named entities.

    Returns:
        List[Tuple[str, str]]: A list of tuples where the first element of each tuple is a named entity from 
        the corpus and the second element is the type of the named entity.
    """
    processed_corpus = nlp(corpus)
    entities = [(ent.text, ent.label_) for ent in processed_corpus.ents]
    return entities

def latent_dirichlet_allocation(
    corpus: str, 
    n_topics: int = 5
) -> Dict[str, any]:
    """
    Latent Dirichlet Allocation (LDA) is an example of topic model and is used to 
    classify text in a document to a particular topic. It builds a topic per document model and words per 
    topic model, modeled as Dirichlet distributions. This means it represents the document as a distribution
    of topics, instead of a distribution of words.

    Parameters:
        corpus (str): The corpus on which to perform topic modeling.
        n_topics (int, optional): The number of topics to be extracted from the corpus. Defaults to 5.

    Returns:
        Dict[str, any]: A dictionary containing topic distribution and top terms for each topic.
    """
    count_vectorizer = CountVectorizer(stop_words='english')
    corpus_term_matrix = count_vectorizer.fit_transform([corpus])
    
    lda = LDA(n_components=n_topics)
    lda.fit(corpus_term_matrix)
    
    topic_distribution  = lda.transform(corpus_term_matrix)[0]
    
    words = count_vectorizer.get_feature_names_out()
    topic_terms = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_10_terms = [words[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_terms[f"Topic {topic_idx + 1}"] = top_10_terms

    results = {
        'distribution': topic_distribution,
        'topics': topic_terms
    }
    
    return results

def extract_keywords(
    corpus: str
) -> List[str]:
    """
    This function takes a corpus as input and returns a list of keywords and phrases.
    
    Keyword extraction is a process of extracting the most relevant words and expressions from text. 
    RAKE (Rapid Automatic Keyword Extraction) is a keyword extraction algorithm which sorts words by their 
    degree of importance.

    Parameters:
        corpus (str): The corpus from which to extract keywords.

    Returns:
        List[str]: A list of keywords and phrases extracted from the corpus.
    """
    rake_nltk_var.extract_keywords_from_text(corpus)
    return rake_nltk_var.get_ranked_phrases()

def feature_engineering_pipeline(
    sentences: List[str]
) -> Dict[str, any]:
    """
    This function takes a list of sentences as input and returns a dictionary of various 
    features extracted from the sentences.

    Parameters:
        sentences (List[str]): The list of sentences from which to extract features.

    Returns:
        Dict[str, any]: A dictionary where the keys are the names of the features and the values are the 
        extracted features.
    """
    corpus = " ".join(sentences)
    features = {}
    
    features['tfidf'] = extract_tfidf_features(sentences)
    features['embeddings'] = extract_sentence_embeddings(sentences)
    features['entities'] = extract_named_entities(corpus)
    features['topics'] = latent_dirichlet_allocation(corpus)
    features['keywords'] = extract_keywords(corpus)
    
    return features

def feature_engineering_pipeline_dataframe(
    df: pd.DataFrame,
    corpus_column: str
) -> Dict[str, List[Any]]:
    """
    Apply feature engineering pipeline to a DataFrame.This function processes 
    each row of the input DataFrame, applying various natural language processing 
    techniques to extract features from the text in the specified corpus column.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        corpus_column (str): Name of the column in the DataFrame that contains
                             the text corpus to be processed.

    Returns:
        Dict[str, List[Any]]: A dictionary containing lists of extracted features.
            The keys of the dictionary are:
            - 'tfidf': TF-IDF vectors for each document
            - 'embeddings': Document embeddings
            - 'entities': Named entities extracted from each document
            - 'topics': Topic assignments for each document
            - 'keywords': Key phrases or words extracted from each document

    Raises:
        KeyError: If the specified corpus_column is not found in the DataFrame.
        ValueError: If the DataFrame is empty.
    """
    features: Dict[str, List[Any]] = {
        'tfidf': [],
        'embeddings': [],
        'entities': [],
        'topics': [],
        'keywords': []
    }
    
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    
    if corpus_column not in df.columns:
        raise KeyError(f"Column '{corpus_column}' not found in the DataFrame.")
    
    # Iterate through each row in the DataFrame using tqdm for progress tracking
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting features"):
        corpus: str = row[corpus_column]
        sentences: List[str] = nltk.sent_tokenize(corpus)
        row_features: Dict[str, Any] = feature_engineering_pipeline(sentences)
        
        # Append the extracted features for each feature type
        for key in features.keys():
            features[key].append(row_features[key])
    
    return features

# Example usage
if __name__ == "__main__":
    
    import pandas as pd
    from pprint import pprint
    
    # Sample DataFrame
    sample_data = [
        {"id": 1, "text": "Climate change is a pressing global issue. Rising temperatures and extreme weather events are affecting ecosystems worldwide."},
        {"id": 2, "text": "Renewable energy sources like solar and wind power are becoming increasingly important in the fight against climate change."},
        {"id": 3, "text": "Deforestation contributes significantly to global warming. Protecting and restoring forests is crucial for mitigating climate change."}
    ]
    df = pd.DataFrame(sample_data)

    # Apply the feature engineering pipeline to the DataFrame
    try:
        features = feature_engineering_pipeline_dataframe(df, corpus_column="text")
        
        # Print out some of the extracted features
        print("Example of extracted features for the first document:")
        print("\nTF-IDF scores (top 5):")
        pprint(dict(sorted(features['tfidf'][0].items(), key=lambda x: x[1], reverse=True)[:5]))
        
        print("\nEmbeddings (shape):")
        print(features['embeddings'][0].shape)
        
        print("\nNamed Entities:")
        pprint(features['entities'][0][:5]) # print first 5 entities
        
        print("\nTopic Distribution:")
        pprint(features['topics'][0]['distribution'])
        
        print("\nKeywords:")
        pprint(features['keywords'][0][:5]) # print first 5 keywords

    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nExample of using individual functions:")
    
    sample_sentences = [
        "The Intergovernmental Panel on Climate Change (IPCC) provides regular assessments on climate change.",
        "These reports are crucial for informing policy decisions worldwide."
    ]
    
    print("\nSentence Embeddings:")
    embeddings = extract_sentence_embeddings(sample_sentences)
    print(f"Shape of embeddings: {embeddings.shape}")
    
    print("\nNamed Entities:")
    entities = extract_named_entities(" ".join(sample_sentences))
    pprint(entities)

    print("\nTF-IDF Scores (top 5):")
    tfidf_scores = extract_tfidf_features(sample_sentences)
    pprint(dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5]))