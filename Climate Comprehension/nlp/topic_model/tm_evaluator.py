import numpy as np
from typing import List, Dict, Any
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from topic_model.tm_config import EvaluationConfig

class TopicModelEvaluator:
    """
    A class for evaluating topic models, specifically designed to compare
    CLIMATopic with standard BERTopic on climate change articles.
    """

    def __init__(
        self, 
        config: EvaluationConfig
    ):
        """
        Initialize the TopicModelEvaluator with the given configuration.

        Args:
            config (EvaluationConfig): Configuration for the evaluation process.
        """
        self.config = config

    def evaluate_coherence(
        self,
        model: BERTopic, 
        docs: List[str]
    ) -> float:
        """
        Evaluate the coherence of a given topic model.

        This method calculates the coherence score of the topic model using the
        specified coherence measure. Higher coherence scores generally indicate
        better topic quality and interpretability.

        Args:
            model (BERTopic): The topic model to evaluate.
            docs (List[str]): The list of documents used to train the model.

        Returns:
            float: The coherence score of the model.
        """
        # Prepare the data for coherence calculation
        vectorizer = CountVectorizer()
        bow_corpus = vectorizer.fit_transform(docs)
        id2word = {id: word for word, id in vectorizer.vocabulary_.items()}

        # Get the top words for each topic
        topics = model.get_topics()
        topic_words = [[word for word, _ in topic] for topic in topics.values()]

        # Calculate coherence
        cm = CoherenceModel(
            topics=topic_words,
            texts=[doc.split() for doc in docs],
            dictionary=Dictionary.from_corpus(bow_corpus, id2word),
            coherence=self.config.coherence_measure,
            topn=self.config.top_n_words
        )
        return cm.get_coherence()

    def evaluate_topic_diversity(
        self,
        model: BERTopic
    ) -> float:
        """
        Evaluate the diversity of topics in a given topic model.

        This method calculates the average pairwise cosine distance between topic vectors.
        Higher diversity scores indicate that the topics are more distinct from each other.

        Args:
            model (BERTopic): The topic model to evaluate.

        Returns:
            float: The topic diversity score.
        """
        topics = model.get_topics()
        topic_vectors = [
            [weight for _, weight in sorted(topic, key=lambda x: x[0])[:self.config.num_words]]
            for topic in topics.values()
        ]
        topic_vectors = np.array(topic_vectors)
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(topic_vectors)
        
        # Convert similarities to distances and calculate average
        distances = 1 - similarities
        diversity = np.mean(distances[np.triu_indices(len(topics), k=1)])
        
        return diversity

    def evaluate_model(
        self, 
        model: BERTopic, 
        docs: List[str]
    ) -> Dict[str, float]:
        """
        Perform a comprehensive evaluation of a topic model.

        This method calculates multiple evaluation metrics for the given topic model,
        including coherence and topic diversity.

        Args:
            model (BERTopic): The topic model to evaluate.
            docs (List[str]): The list of documents used to train the model.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation results.
        """
        coherence = self.evaluate_coherence(model, docs)
        diversity = self.evaluate_topic_diversity(model)
        
        return {
            "coherence": coherence,
            "diversity": diversity
        }

    def evaluate_single_document(
        self, 
        model: BERTopic, doc: str
    ) -> Dict[str, Any]:
        """
        Evaluate the performance of a topic model on a single document.

        This method analyzes the topics assigned to a single document and provides
        insights into the model's performance on that specific text.

        Args:
            model (BERTopic): The topic model to evaluate.
            doc (str): The document to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for the document.
        """
        # Extract topics for the document
        topics, probs = model.transform([doc])
        
        # Get the top topic and its probability
        top_topic = topics[0]
        top_prob = probs[0][np.argmax(probs[0])]
        
        # Get the words for the top topic
        topic_words = model.get_topic(top_topic)
        
        return {
            "top_topic": top_topic,
            "top_probability": top_prob,
            "topic_words": topic_words
        }

    @staticmethod
    def visualize_comparison(
        results: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Visualize the comparison between two topic models.

        This method creates a bar plot comparing the coherence and diversity
        scores of two topic models (e.g., CLIMATopic and standard BERTopic).

        Args:
            results (Dict[str, Dict[str, float]]): A dictionary containing the
                evaluation results for each model.
        """
        metrics = list(next(iter(results.values())).keys())
        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = sns.color_palette("Set2")
        for i, (model_name, model_results) in enumerate(results.items()):
            ax.bar(x + i*width, list(model_results.values()), width, label=model_name, color=colors[i])

        ax.set_ylabel('Scores', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_title('Comparison of CLIMATopic and BERTopic Performance', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=10)

        for i, v in enumerate(list(results.values())[0].values()):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_single_document_comparison(
        climatopic_result: Dict[str, Any], 
        bertopic_result: Dict[str, Any]
    ) -> None:
        """
        Visualize the comparison of CLIMATopic and BERTopic on a single document.

        This method creates a plot comparing the top topics and their probabilities
        for both models on a single document.

        Args:
            climatopic_result (Dict[str, Any]): The evaluation result for CLIMATopic on the single document.
            bertopic_result (Dict[str, Any]): The evaluation result for BERTopic on the single document.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot top topic probabilities
        models = ['CLIMATopic', 'BERTopic']
        probabilities = [climatopic_result['top_probability'], bertopic_result['top_probability']]
        colors = sns.color_palette("Set2")[:2]
        
        ax1.bar(models, probabilities, color=colors)
        ax1.set_ylabel('Top Topic Probability', fontsize=12)
        ax1.set_title('Top Topic Probability Comparison\non Single Complex Document', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(probabilities):
            ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')

        # Plot top topic words
        climatopic_words = [word for word, _ in climatopic_result['topic_words'][:10]]
        bertopic_words = [word for word, _ in bertopic_result['topic_words'][:10]]
        
        ax2.barh(range(10), [1]*10, tick_label=climatopic_words, alpha=0.6, color=colors[0], label='CLIMATopic')
        ax2.barh(range(10), [0.5]*10, tick_label=bertopic_words, alpha=0.6, color=colors[1], label='BERTopic')
        ax2.set_title('Top 10 Topic Words Comparison\non Single Complex Document', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Relative Importance', fontsize=12)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(climatopic_words, fontsize=10)
        ax2.legend(fontsize=10, loc='lower right')

        plt.tight_layout()
        plt.show()

def evaluate_topic_models(
    climatopic: BERTopic, 
    bertopic: BERTopic, 
    docs: List[str], 
    single_doc: str, 
    config: EvaluationConfig
) -> Dict[str, Any]:
    """
    Evaluate and compare CLIMATopic and standard BERTopic models.

    This function performs a comprehensive evaluation of both models on a set of documents
    and a single complex document, returning the results and visualizations.

    Args:
        climatopic (BERTopic): The pre-trained CLIMATopic model to evaluate.
        bertopic (BERTopic): The standard BERTopic model to evaluate.
        docs (List[str]): The list of documents used to evaluate the models.
        single_doc (str): A single complex document for detailed evaluation.
        config (EvaluationConfig): Configuration for the evaluation process.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results and comparison visualizations.
    """
    evaluator = TopicModelEvaluator(config)
    
    # Evaluate on multiple documents
    climatopic_topics, _ = climatopic.transform(docs)
    bertopic_topics, _ = bertopic.fit_transform(docs)
    
    climatopic_results = evaluator.evaluate_model(climatopic, docs)
    bertopic_results = evaluator.evaluate_model(bertopic, docs)
    
    results = {
        "CLIMATopic": climatopic_results,
        "BERTopic": bertopic_results
    }
    
    # Evaluate on single document
    climatopic_single = evaluator.evaluate_single_document(climatopic, single_doc)
    bertopic_single = evaluator.evaluate_single_document(bertopic, single_doc)
    
    results["SingleDocument"] = {
        "CLIMATopic": climatopic_single,
        "BERTopic": bertopic_single
    }
    
    # Visualize results
    TopicModelEvaluator.visualize_comparison(results)
    TopicModelEvaluator.visualize_single_document_comparison(climatopic_single, bertopic_single)
    
    return results
    
if __name__ == "__main__":
    from bertopic import BERTopic
    from topic_model.tm_config import EvaluationConfig

    # Sample data
    docs = [
        "Climate change is affecting global temperatures.",
        "Renewable energy is crucial for sustainability.",
        "Carbon emissions are a major concern for the environment.",
        "Sea levels are rising due to global warming.",
        "Sustainable practices can help mitigate climate change."
    ]
    single_doc = "The Intergovernmental Panel on Climate Change (IPCC) report highlights the urgent need for global action to reduce greenhouse gas emissions and adapt to the impacts of climate change."

    climatopic = BERTopic()
    bertopic = BERTopic()
    
    climatopic.fit(docs)
    bertopic.fit(docs)
    
    config = EvaluationConfig()
    
    results = evaluate_topic_models(climatopic, bertopic, docs, single_doc, config)
    
    print("Evaluation Results:")
    for model, metrics in results.items():
        if model != "SingleDocument":
            print(f"{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    print("\nSingle Document Evaluation:")
    for model, result in results["SingleDocument"].items():
        print(f"{model}:")
        print(f"  Top Topic: {result['top_topic']}")
        print(f"  Top Probability: {result['top_probability']}")
        print(f"  Top 5 Topic Words: {result['topic_words'][:5]}")
    
    print("\nVisualization plots have been displayed.")