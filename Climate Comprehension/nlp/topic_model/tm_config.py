from dataclasses import dataclass
from typing import Tuple, List, Union

@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 30
    min_samples: int = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom" # excess of mass
    prediction_data: bool = True

@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 10

@dataclass
class CountVectorizerConfig:
    stop_words: str = "english"
    min_df: int = 2
    ngram_range: Tuple[int, int] = (1, 2)

@dataclass
class KeyBERTConfig:
    top_n_words: int = 10
    random_state: int = 10

@dataclass
class MMRConfig:
    diversity: float = 0.5

@dataclass
class BERTopicConfig:
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_n_words: int = 10
    verbose: bool = True
    seed_topics: List[str] = (
        "Climate Models", "Paleoclimatology", "Meteorology", "Oceanography", 
        "Glaciology", "Climate Feedback Mechanisms", "Atmospheric Chemistry",
        "Biodiversity Loss", "Habitat Destruction", "Ecosystem Services", 
        "Ocean Acidification", "Soil Degradation", "Water Scarcity", 
        "Natural Disasters", "Air Quality", "Public Health", "Disease Spread", 
        "Heatwaves", "Renewable Energy", "Fossil Fuels", "Carbon Capture and Storage",
        "Energy Efficiency", "Nuclear Energy", "Biofuels", "Greenhouse Gas Emissions",
        "International Agreements", "National Policies and Legislation", "Carbon Pricing",
        "Environmental Law", "Climate Policy Analysis", "Climate Economics", 
        "Climate Justice", "Social Impact and Vulnerability", "Insurance and Risk Management",
        "Migration and Displacement", "Green Jobs and Economy", "Sustainable Practices", 
        "Resilience Planning", "Disaster Management", "Urban Planning", 
        "Agricultural Adaptation", "Water Management", "Green Technology", 
        "Innovations in Sustainability", "Technological Advancements in Mitigation", 
        "AI and Machine Learning in Climate Science", "Geoengineering", 
        "Climate Education", "Media Coverage", "Public Opinion", 
        "Climate Change Denial and Conspiracies", "Misinformation and Fake News",
        "Arctic and Antarctic Ice Melt", "Coral Bleaching", "Extreme Weather Events",
        "Sea Level Rise", "Droughts", "Forest Fires"
    )
    nr_topics: Union[str, int] = "auto"
    
@dataclass
class SaveConfig:
    path: str = "CLIMATopic"
    serialization: str = "safetensors"
    save_ctfidf: bool = True
    save_embedding_model: str = "all-MiniLM-L6-v2"
    
@dataclass
class VisualizationConfig:
    width: int = 1200
    height: int = 1200
    title: str = "Documents and Topics"
    sub_title: Union[str, None] = None
    
@dataclass
class EvaluationConfig:
    num_topics: int = 10
    coherence_measure: str = "c_v"  # coherence measure to use
    top_n_words: int = 10  # number of top words to consider for coherence calculation
    num_words: int = 30  # number of words to consider for topic diversity
    random_state: int = 10
    num_articles: int = 100