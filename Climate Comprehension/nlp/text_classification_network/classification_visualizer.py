import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple

class SphericalMap:
    """
    Initializes the SphericalMap object.
    
        Attributes:
        label_encoder (LabelEncoder): An instance of sklearn's LabelEncoder for encoding labels.
        umap_model (UMAP or None): UMAP model for dimensionality reduction, initially set to None.
        encoded_labels (array or None): Encoded version of the labels, initially set to None.
        embedding (array or None): UMAP embedding of the data, initially set to None.
        original_labels (array or None): Original labels before encoding, initially set to None.
        topics (array or None): Assigned topics for each data point, initially set to None.
    """
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.umap_model = None
        self.encoded_labels = None
        self.embedding = None
        self.original_labels = None
        self.topics = None

    def encode_labels(
        self, 
        labels: List[str]
    ) -> np.ndarray:
        """
        Encode categorical labels to numerical values.

        Args:
            labels (List[str]): List of categorical labels.

        Returns:
            np.ndarray: Array of encoded numerical labels.
        """
        self.original_labels = labels
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        return self.encoded_labels

    def fit_umap(
        self, 
        data: np.ndarray, 
        topics: List[str], 
        n_neighbors: int = 15, 
        min_dist: float = 0.1, 
        n_components: int = 3, 
        random_state: int = 42
    ) -> np.ndarray:
        """
        Fit UMAP model to the input data.

        Args:
            data (np.ndarray): Input data to fit UMAP.
            topics (List[str]): List of topics corresponding to each data point.
            n_neighbors (int): Number of neighbors for UMAP. Default is 15.
            min_dist (float): Minimum distance for UMAP. Default is 0.1.
            n_components (int): Number of components for UMAP. Default is 3.
            random_state (int): Random state for reproducibility. Default is 42.

        Returns:
            np.ndarray: UMAP embedding of the input data.
        """
        self.topics = topics
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            metric='cosine'
        )
        self.embedding = self.umap_model.fit_transform(data)
        return self.embedding

    def convert_to_sphere(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert UMAP embedding to 3D coordinates on a unit sphere.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and z coordinates on the unit sphere.
        """
        # Normalize to unit sphere
        norms = np.sqrt((self.embedding**2).sum(axis=1))
        x = self.embedding[:, 0] / norms
        y = self.embedding[:, 1] / norms
        z = self.embedding[:, 2] / norms
        return x, y, z

    def visualize(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray, 
        labels: np.ndarray, 
        title: str = "Spherical Map Distribution"
    ) -> None:
        """
        Create and display a 3D scatter plot of the data on a sphere.

        Args:
            x (np.ndarray): x-coordinates of the data points.
            y (np.ndarray): y-coordinates of the data points.
            z (np.ndarray): z-coordinates of the data points.
            labels (np.ndarray): Encoded labels for coloring the points.
            title (str): Title of the plot. Default is "Spherical Map Visualization".
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a color map for classification labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot points for each classification label
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(x[mask], y[mask], z[mask], c=[color], 
                       label=self.label_encoder.inverse_transform([label])[0], alpha=0.7)
        
        # Add topic labels
        for i, topic in enumerate(self.topics):
            ax.text(x[i], y[i], z[i], topic, fontsize=8, alpha=0.7)
        
        ax.set_title(title)
        ax.legend(title="Classifications", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def create_visualization(
        self, 
        data: np.ndarray, 
        labels: List[str], 
        topics: List[str]
    ) -> None:
        """
        Create a full visualization pipeline: encode labels, fit UMAP, convert to sphere, and visualize.

        Args:
            data (np.ndarray): Input data for visualization.
            labels (List[str]): List of categorical labels.
            topics (List[str]): List of topics corresponding to each data point.
        """
        encoded_labels = self.encode_labels(labels)
        self.fit_umap(data, topics)
        x, y, z = self.convert_to_sphere()
        self.visualize(x, y, z, encoded_labels)

# Example usage
if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Sample data
    df = pd.DataFrame({
        'corpus': ['Climate change impact', 'Ocean acidification study', 'Renewable energy policy'],
        'classification': ['Research Article', 'Review', 'Policy Brief'],
        'topic': ['Climate Science', 'Oceanography', 'Energy Policy']
    })

    # Create an instance of SphericalMapVisualizer
    visualizer = SphericalMap()

    # Prepare data using TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['corpus']).toarray()

    # Create the visualization
    visualizer.create_visualization(X, df['classification'].tolist(), df['topic'].tolist())