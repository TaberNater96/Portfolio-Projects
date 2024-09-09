import nltk
from nltk.data import find
from typing import List, Tuple

def download_nltk_resources() -> None:
    """
    Ensure that necessary NLTK resources are downloaded.

    This function checks for the existence of specific NLTK resources required 
    for the project. If any resource is missing, it downloads the resource. 
    This helps in setting up the environment without redundant downloads.

    Resources checked:
    - stopwords
    - wordnet
    - punkt
    - averaged_perceptron_tagger
    """
    # List of resources to check and download if necessary from NLTK
    resources: List[Tuple[str, str]] = [
        ('corpora/stopwords.zip', 'stopwords'),
        ('corpora/wordnet.zip', 'wordnet'),
        ('corpora/punkt.zip', 'punkt')
        ('taggers/averaged_perceptron_tagger.zip', 'averaged_perceptron_tagger'),
    ]

    # Check each resource and download if necessary
    for resource_path, resource_name in resources:
        try:
            find(resource_path)
            print(f"{resource_name} is already downloaded.")
        except LookupError:
            print(f"Downloading {resource_name}...")
            nltk.download(resource_name)