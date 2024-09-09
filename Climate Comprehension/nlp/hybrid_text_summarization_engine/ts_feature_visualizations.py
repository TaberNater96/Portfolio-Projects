import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import umap.plot
from wordcloud import WordCloud
from typing import Dict, List, Any
import networkx as nx

def visualize_tfidf(
    tfidf_scores: Dict[str, float], 
    top_n: int = 20
):
    """
    Visualize TF-IDF scores using a horizontal bar plot.

    Args:
        tfidf_scores (Dict[str, float]): Dictionary of words and their TF-IDF scores.
        top_n (int): Number of top terms to display.
    """
    sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, scores = zip(*sorted_scores)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(scores), y=list(words), orient='h')
    plt.title(f'Top {top_n} Terms by TF-IDF Score')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Terms')
    plt.tight_layout()
    plt.show()

def visualize_embeddings(embeddings: np.ndarray):
    """
    Visualize sentence embeddings using UMAP.

    Args:
        embeddings (np.ndarray): Array of sentence embeddings.
    """
    reducer = umap.UMAP()
    reducer.fit(embeddings)

    plt.figure(figsize=(12, 8))
    umap.plot.points(reducer, labels=np.arange(len(embeddings)), theme='fire')
    plt.title('UMAP Visualization of Sentence Embeddings')
    plt.tight_layout()
    plt.show()

def visualize_topics(topic_data: Dict[str, Any]):
    """
    Visualize LDA topic modeling results using a heatmap and bar charts.

    Args:
        topic_data (Dict[str, Any]): Dictionary containing topic distribution and top terms.
    """
    distribution = topic_data['distribution']
    topics = topic_data['topics']

    # Heatmap of topic distribution
    plt.figure(figsize=(12, 4))
    sns.heatmap([distribution], cmap='seismic', annot=True, cbar=False)
    plt.title('Topic Distribution')
    plt.xlabel('Topics')
    plt.tight_layout()
    plt.show()

    # Bar charts for top terms in each topic
    n_topics = len(topics)
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 4*n_topics))
    for i, (topic, terms) in enumerate(topics.items()):
        term_weights = [1/(j+1) for j in range(len(terms))]  # simple weighting
        sns.barplot(x=term_weights, y=terms, ax=axes[i])
        axes[i].set_title(f'{topic} - Top Terms')
        axes[i].set_xlabel('Weight')
    plt.tight_layout()
    plt.show()

def visualize_named_entities(entities: List[tuple]):
    """
    Visualize named entities using a network graph.

    Args:
        entities (List[tuple]): List of (entity, type) tuples.
    """
    G = nx.Graph()
    for entity, entity_type in entities:
        G.add_node(entity, type=entity_type)
        G.add_edge(entity, entity_type)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(G, pos)
    plt.title('Named Entity Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_keywords(keywords: List[str]):
    """
    Visualize keywords using a word cloud and a bar chart.

    Args:
        keywords (List[str]): List of extracted keywords.
    """
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keyword Word Cloud')
    plt.tight_layout()
    plt.show()

    # Bar Chart of top keywords
    keyword_freq = pd.Series(keywords).value_counts().head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=keyword_freq.values, y=keyword_freq.index)
    plt.title('Top 20 Keywords')
    plt.xlabel('Frequency')
    plt.ylabel('Keywords')
    plt.tight_layout()
    plt.show()