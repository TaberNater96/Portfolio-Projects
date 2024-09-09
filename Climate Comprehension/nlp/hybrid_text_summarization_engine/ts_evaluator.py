import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple ,Any
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
from nltk import word_tokenize
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm

class BARTevaluator:
    """
    A class for evaluating the performance of the climate corpus condenser (C3) and a standard
    BART model. This class provides methods to compute various metrics for evaluating the performance
    of BART models, including ROUGE scores, METEOR scores, a confusion matrix, and other custom metrics.
    """
    
    def __init__(
        self, 
        c3_model: Any, 
        bart_model: Any
    ):
        """
        Initialize the SummarizationEvaluator with both the C3 and standard BART. The rouge scorer
        is a package that provides a convenient way to compute the ROUGE scores between two strings.
        The main class is designed to validate through n-gram based scoring (rouge1, ... rougen), and 
        longest common subsequence (rougeL). The Porter Stemmer Algorithm is a process that removes
        morphological and inflexional endings from words, which is necessart for matching similar words. 
        Here, it will default to True.
        
        Args:
            c3_model (Any): Climate Corpus Condenser (C3) model.
            bart_model (Any): BART model.
        """
        self.c3_model = c3_model
        self.bart_model = bart_model
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rouge3'], # measures overlap of unigrams, bigrams, and trigrams
            use_stemmer=True # bool to indicate if the Porter stemmer should be used strip suffixes
        )
        
    def compute_rouge_scores(
        self,
        reference: str,
        canidate: str
    ) -> Dict[str, float]:
        """
        Compute the ROUGE scores for the reference and candidate summaries (i.e. between the target
        and prediction). The f-measure is taken from each rouge metric and returned as a dictionary. 
        The f-measure (essentially an F1 score) is the harmonic mean of precision and recall and gives 
        an overall measure of the quality of the text comparison.
        
        Args:
            reference (str): The reference (ground truth) summary.
            canidate (str): The candidate (generated) summary.
            
        Returns:
            Dict[str, float]: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1-scores.
        """
        scores = self.rouge_scorer.score(reference, canidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        
    def compute_meteor_score(
        self, 
        reference: str, 
        canidate: str
    ) -> float:
        """
        METEOR: Metric for Evaluation of Translation with Explicit ORdering
        
        Computes the METEOR score between the reference and candidate summaries. Unigrams can be 
        matched  bases on their surface forms, stemmed forms,and meanings. Once all generalized unigram 
        matches between the two strings have been found, METEOR computs a score for this matching using 
        a combination of unigram-precision, recall, and framgmentation, which is designed to capture how 
        well-ordered the matched words in the machine translation are in relation to the reference.
        
        Args:
            reference (str): The reference (ground truth) summary.
            canidate (str): The candidate (generated) summary.
            
        Returns:
            float: The METEOR score.
        """
        return meteor_score.meteor_score(
            [word_tokenize(reference)],
            word_tokenize(canidate)
        )
        
    def compute_length_ratio(
        self,
        reference: str,
        canidate: str
    ) -> float:
        """
        Computes the length ratio between the reference and candidate summaries. In a typical Brevity Penalty 
        (BP) if this ratio is less than 1 (meaning the canidate is shorter than the reference), a penalty is 
        applied. This method is a simpler version of the BP, only addressing how close in length the two
        summaries are, allowing for shorter and longer summaries to be treated equally.
        
        Args:
            reference (str): The reference (ground truth) summary.
            canidate (str): The candidate (generated) summary.
            
        Returns:
            float: The length ratio (candidate length / reference length).
        """
        return len(canidate.split()) / len(reference.split())
    
    def compute_novelty_score(
        self, 
        article: str,
        summary: str
    ) -> float:
        """
        Computes the novelty score of the summary compared to the original article. A novelty score is a 
        metric used to evaluate how much new or unique content is introduced in a generated summary compared 
        to the original text. Essentially, it measures the proportion of words in the summary that do not
        appear in the original article.
        
        Args:
            article (str): The original article text.
            summary (str): The generated summary text.
            
        Returns:
            float: The novelty score (proportion of novel words in the summary).
        """
        
        # First create a set of novel words through word level tokenization for both article and summary
        article_words = set(word_tokenize(article.lower()))
        summary_words = set(word_tokenize(summary.lower()))
        
        # Find the set of novel words
        novel_words = summary_words - article_words
        return len(novel_words) / len(summary_words) if summary_words else 0 # avoids division by zero error
    
    def evaluate_model(
        self,
        model: Any,
        articles: List[str],
        reference_summaries: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates a BART summarization model on unseen evaluation articles and reference summaries. This 
        method puts the full pipeline together to compute the various evaluation metrics defined previously.
        Each metric is computed for each article, which are then stored in a list. The results for all scores
        for each article are then averaged to obtain the final evaluation score.
        
        Args:
            model (Any): The BART models to evaluate.
            articels (List[str]): The list of input articles for evaluation (separte from training data).
            reference_summaries (List[str]): The list of reference summaries for evaluation.
        
        Returns:
            Dict[str, Any]: A dictionary containing the various evaluation results.
        """
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        meteor_scores = []
        length_ratios = []
        novelty_scores = []
        
        for article, reference in tqdm(zip(articles, reference_summaries), 
                                       total=len(articles), 
                                       desc="Evaluating"):
            candidate = model.summarize(article)
            rouge = self.compute_rouge_scores(reference, candidate)
            
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key])
                
            meteor_scores.append(self.compute_meteor_score(reference, candidate))
            length_ratios.append(self.compute_length_ratio(reference, candidate))
            novelty_scores.append(self.compute_novelty_score(article, candidate))
            
        return {
            'rouge': {k: np.mean(v) for k,v in rouge_scores.items()},
            'meteor': np.mean(meteor_scores),
            'length_ratio': np.mean(length_ratios),
            'novelty': np.mean(novelty_scores)
        }
            
    def compare_models(
        self,
        articles: List[str],
        reference_summaries: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compares the C3 and standard BART models on unseen evaluation articles along with their respective 
        reference summaries. Sequentially activate both evaluation pipelines for each model.
        
        Args: 
            articles (List[str]): List of input articles for evaluation.
            reference_summaries (List[str]): List of reference summaries for evaluation.
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Evaluation results for C3 and BART models.
        """
        c3_results = self.evaluate_model(self.c3_model, articles, reference_summaries)
        bart_results = self.evaluate_model(self.bart_model, articles, reference_summaries)
        
        return c3_results, bart_results
    
    def visualize_results(
        self, 
        c3_results: Dict[str, Any], 
        bart_results: Dict[str, Any]
    ) -> None:
        """
        Visualize the comparison results between C3 and BART models.
        
        Args:
            c3_results (Dict[str, Any]): Evaluation results for the C3 model.
            bart_results (Dict[str, Any]): Evaluation results for the BART model.
        """
        metrics = ['rouge1', 'rouge2', 'rougeL', 'meteor', 'length_ratio', 'novelty']
        
        c3_scores = [
            c3_results['rouge']['rouge1'], 
            c3_results['rouge']['rouge2'], 
            c3_results['rouge']['rougeL'],
            c3_results['meteor'], 
            c3_results['length_ratio'], 
            c3_results['novelty']
        ]
        bart_scores = [
            bart_results['rouge']['rouge1'], 
            bart_results['rouge']['rouge2'], 
            bart_results['rouge']['rougeL'],
            bart_results['meteor'], 
            bart_results['length_ratio'], 
            bart_results['novelty']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, c3_scores, width, label='C3 Model')
        ax.bar(x + width/2, bart_scores, width, label='BART Model')
        
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of C3 and BART Models')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add value labels on top of each bar
        for i, v in enumerate(c3_scores):
            ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(bart_scores):
            ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
        
        fig.tight_layout()
        plt.show()
        
    def generate_summary_wordcloud(
        self,
        summaries: List[str]
    ) -> None:
        """
        Generate and display a word cloud from the list of summaries.
        
        Args:
            summaries (List[str]): List of generated summaries to visualize.
        """
        text = ' '.join(summaries)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Generated Summaries')
        plt.show()
        
    def analyze_topic_coverage(
        self,
        articles: List[str],
        summaries: List[str]
    ) -> None:
        """
        Analyze and visualize the topic coverage in summaries compared to original articles.
        
        Args:
            articles (List[str]): List of original articles.
            summaries (List[str]): List of generated summaries.
        """
        def extract_key_terms(
            text: str, 
            n: int = 10
        ) -> List[str]:
            """Extract the most frequently used words in a summary."""
            words = word_tokenize(text.lower())
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(n)]
        
        # Create a nested list of key terms for each article and summary
        article_terms = [extract_key_terms(article) for article in articles]
        summary_terms = [extract_key_terms(summary) for summary in summaries]
        
        # Create a list of topic coverage scores
        coverage_scores = []
        for art_terms, sum_terms in zip(article_terms, summary_terms):
            coverage = len(set(art_terms) & set(sum_terms)) / len(set(art_terms))
            coverage_scores.append(coverage)
            
        plt.figure(figsize=(10, 5))
        sns.histplot(coverage_scores, kde=True)
        plt.title('Distribution of Topic Coverage Scores')
        plt.xlabel('Coverage Score')
        plt.ylabel('Frequency')
        plt.show()