import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
from bertopic import BERTopic
from typing import List, Union

def visualize_umap(embeddings: np.ndarray, umap_model: umap.UMAP):
    """
    Creates a UMAP (Uniform Manifold Approximation and Projection) visualization of document embeddings.

    This function fits the UMAP model to the provided embeddings and generates a plot that:
    1. Reduces the high-dimensional embeddings to a 2D representation.
    2. Visualizes the connectivity between data points using edge bundling.
    3. Helps identify clusters and patterns in the document space.

    Args:
        embeddings (np.ndarray): A 2D numpy array where each row represents a document's embedding.
        umap_model (umap.UMAP): A configured UMAP model instance to use for dimensionality reduction.
    """
    mapper = umap_model.fit(embeddings)
    umap.plot.connectivity(
        mapper, 
        edge_bundling='hammer'
    )
    plt.show()

def visualize_topics(topic_model: BERTopic):
    """
    Generates a visualization of the topics discovered by BERTopic.

    This function utilizes BERTopic's built-in visualization capabilities to create
    an interactive plot that displays:
    1. The most relevant words for each topic.
    2. The relative size of each topic (based on the number of documents assigned to it).
    3. The relationships between topics (proximity in the visualization often indicates similarity).

    Args:
        topic_model (BERTopic): A fitted BERTopic model containing discovered topics.
    """
    topic_model.visualize_topics()

def visualize_heatmap(topic_model: BERTopic):
    """
    Creates a heatmap visualization of topic similarities.

    This function generates a color-coded matrix where each cell represents
    the similarity between two topics. The heatmap helps to:
    1. Identify clusters of related topics.
    2. Spot potential redundancies in the topic model.
    3. Understand the overall structure of the topic space.

    Args:
        topic_model (BERTopic): A fitted BERTopic model containing discovered topics.
    """
    topic_model.visualize_heatmap(title="CLIMATopic Similarity Matrix", width=1000, height=1000)

def visualize_document_datamap(topic_model: BERTopic, docs: List[str], topics: List[int] = None,
                               embeddings: np.ndarray = None, reduced_embeddings: np.ndarray = None,
                               custom_labels: Union[bool, str] = False, title: str = "Documents and Topics",
                               sub_title: Union[str, None] = None, width: int = 1200, height: int = 1200,
                               **datamap_kwds):
    """
    Creates a 2D scatter plot visualization of documents and their assigned topics.

    This function generates an interactive plot where each point represents a document,
    and the color of the point indicates its assigned topic. The plot provides insights into:
    1. The distribution of documents across different topics.
    2. The relationships between documents and their proximity in the semantic space.
    3. The potential overlap or distinction between different topics.

    Args:
        topic_model (BERTopic): A fitted BERTopic model.
        docs (List[str]): List of document texts.
        topics (List[int], optional): List of topic assignments for each document.
        embeddings (np.ndarray, optional): Document embeddings.
        reduced_embeddings (np.ndarray, optional): Reduced-dimensional embeddings (e.g., from UMAP).
        custom_labels (Union[bool, str], optional): Custom labels for the documents.
        title (str): Title of the plot.
        sub_title (Union[str, None], optional): Subtitle of the plot.
        width (int): Width of the plot in pixels.
        height (int): Height of the plot in pixels.
        **datamap_kwds: Additional keyword arguments for customizing the plot.
    """
    fig = topic_model.visualize_document_datamap(docs, topics=topics, embeddings=embeddings,
                                                 reduced_embeddings=reduced_embeddings,
                                                 custom_labels=custom_labels, title=title,
                                                 sub_title=sub_title, width=width, height=height,
                                                 **datamap_kwds)
    fig.show()
    
if __name__ == "__main__":
    import numpy as np
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # Sample data
    docs = [
        "Climate change is affecting global temperatures.",
        "Renewable energy is crucial for sustainability.",
        "Carbon emissions are a major concern for the environment.",
        "Sea levels are rising due to global warming.",
        "Sustainable practices can help mitigate climate change."
    ]

    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(docs)

    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embed_model.encode(docs)

    print("Visualizing UMAP...")
    visualize_umap(embeddings, umap.UMAP())

    print("Visualizing topics...")
    visualize_topics(topic_model)

    print("Visualizing heatmap...")
    visualize_heatmap(topic_model)

    print("Visualizing document datamap...")
    visualize_document_datamap(topic_model, docs, topics, embeddings)

    print("All visualizations have been generated and displayed.")