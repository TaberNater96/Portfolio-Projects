"""
By: Elijah Taber
Dated: May the 4th, 2024

The `ts_corpus_cleaner` script is crafted to refine and prepare raw text for a hybrid text summarization 
model, ideal for processing scientific and technical documents. This tool was specifically crafted for text 
summarization in order to preserve the context of the training corpus, using the power of SpaCy, 
NLTK, and BeautifulSoup to perform essential cleaning tasks. Including stripping HTML, normalizing 
characters, expanding contractions, and enforcing lowercase uniformity. Its unique capability to recognize
and preserve crucial measurements—like kg and cm—using SpaCy's Matcher is pivotal for maintaining data 
integrity in summaries. The `preprocess_text` function fouses these processes into a pipeline 
that ensures the retention of vital information, optimizing the text for precise and effective summarization.
"""

import re
from bs4 import BeautifulSoup
import unicodedata
import spacy
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from ...src.\
        contractions import CONTRACTION_MAP 

# Load SpaCy NLP model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 50_000_000  # adjustable based on text size

# Prepare tokenizer and stopword list
tokenizer = ToktokTokenizer()
stopwords = nltk.corpus.stopwords.words('english')

# Create a Matcher object with the current vocabulary
matcher = Matcher(nlp.vocab)

# Define a list of patterns for the matcher to look for in common measurement units
measurement_patterns = [
    {'LIKE_NUM': True},
    {'LOWER': {'IN': ['kg', 'g', 'tons', 'ton', 'lbs', 'pounds', 'cm', 'm', 'km', 'ft', 'feet', 'inches', 'miles',
                      'ml', 'liters', 'm3', 'gallons', 'gal', '%', 'ppm', 'ppb', '°C', '°F', 'M', 'mol/L', 'J', 
                      'kWh', 'm2', 'square meters', 'acres', 'Pa', 'V', 'A', 'W', 'Bq', 'Ci', 'sec', 'min', 'hr',
                      'Hz', 'degrees', '°', 'rad'
                      ]}}
]


# Add the measurement patterns to the SpaCy matcher under the label "MEASUREMENT"
matcher.add("MEASUREMENT", [measurement_patterns])

def add_measurement_entities(doc: Doc) -> Doc:
    """
    Annotate measurement entities in the document. This function scans the document
    for numerical data followed by specific units (like kg, cm, etc.) and marks them
    as measurement entities to keep for summarization. This is useful for retaining
    relavent measurement values while removing other non-essential numerical data.

    Args:
        doc (spacy.tokens.doc.Doc): The document to process.

    Returns:
        spacy.tokens.doc.Doc: The document with measurement entities added.
    """
    matches = matcher(doc)
    spans = [
        Span(doc, start, end, label="MEASUREMENT")  # create a new Span for each match
        for match_id, start, end in matches  
    ]
    doc.ents = list(doc.ents) + spans  # append new measurement spans to the document's entities
    return doc

def strip_html_tags(text: str) -> str:
    """
    Removes HTML tags and inline JavaScript/CSS from text.

    Args:
        text (str): The input text containing HTML tags and inline JavaScript/CSS.

    Returns:
        str: The text with HTML tags, inline JavaScript, and CSS removed.
    """
    # Create a BeautifulSoup object from the input text using the html.parser, then
    # extract the text from the object while replacing HTML tags with a space
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")

    # Use a regular expression to replace any sequence of whitespace characters
    # with a single space, and then strip leading/trailing whitespace
    return re.sub(r'\s+', ' ', stripped_text).strip()

def remove_accented_chars(text: str) -> str:
    """
    Converts accented characters to ASCII characters.

    Args:
        text (str): The input text containing accented characters.

    Returns:
        str: The text with accented characters converted to ASCII characters.
    """
    # Normalize the input text using the NFKD (Normalization Form KC: Compatibility Composition) 
    # method from the unicodedata module. This method replaces all compatibility characters 
    # in the text with their equivalent composed characters.
    text = unicodedata.normalize('NFKD', text)

    # Encode the normalized text to ASCII, ignoring any errors. Then decode the resulting bytes 
    # back to a string using UTF-8.
    return text.encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_contractions(text: str, contraction_mapping: dict) -> str:
    """
    Expands contractions found in the text using a mapping dictionary with common contractions.

    Args:
        text (str): Text containing contractions to be expanded.
        contraction_mapping (dict): A dictionary where keys are contractions and values are the expanded form.

    Returns:
        str: Text with contractions expanded.
    """
    # Compile a pattern for matching contractions in text
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)

    # Nested function to expand matched contractions
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0] # get the first character of the match
        
        # Expand the contraction using the mapping, or return the match if it's not in the mapping
        expanded_contraction = contraction_mapping.get(match.lower(), match)
        expanded_contraction = first_char + expanded_contraction[1:] # ensure the expanded contraction retains the case of the first character
        return expanded_contraction

    # Expand contractions in the text and remove any single quotes
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text: str, remove_digits: bool = False) -> str:
    """
    Removes special characters, optionally preserving digits.

    Args:
        text (str): The input text containing special characters.
        remove_digits (bool, optional): Whether to remove digits along with special 
        characters. Defaults to False.

    Returns:
        str: The text with special characters removed.
    """
    # Set the pattern to match non-alphanumeric characters and spaces
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    # Use regex to replace matched characters in the text with an empty string, effectively removing them
    return re.sub(pattern, '', text)

def lowercase_text(text: str) -> str:
    """
    Converts all characters in the text to lowercase.

    Args:
        text (str): The input text.

    Returns:
        str: The text with all characters converted to lowercase.
    """
    return text.lower()

def process_and_filter_numbers(doc: Doc) -> Doc:
    """
    This process iterates through the tokens from the input document and filters
    the numbers based on whether they are measurements or years. Measurement numbers
    are determined through NLP entity recognition, while years are determined based 
    on a predefined range of years from 1800 to 2100. This is a workaround for 
    removing non-essential numbers from the corpus while retaining relevant numbers
    without removing all numeric values.

    Args:
        doc (spacy.tokens.Doc): The input document to process.

    Returns:
        spacy.tokens.Doc: The processed document with filtered numbers.
    """
    # List of years from 1800 to 2100
    years = set(range(1800, 2101))
    
    new_tokens = []
    for token in doc:
        if token.like_num:
            # If the token is a measurement entity or a year, append it to the new tokens list
            if token.ent_type_ == "MEASUREMENT" or (token.text.isdigit() and int(token.text) in years):
                new_tokens.append(token)
        else:
            # If the token is not numeric, append it to the new tokens list
            new_tokens.append(token)
            
    # Returns an updated SpaCy Doc object with new tokens, adding spaces between each token
    return spacy.tokens.Doc(nlp.vocab, 
                            words=[token.text for token in new_tokens], 
                            spaces=[True]*len(new_tokens))

def preprocess_text(text: str) -> str:
    """
    Applies all preprocessing functions to the raw corpus in single pipeline.

    Args:
        text (str): The raw text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Initial text processing steps to normalize and remove unwanted elements
    text = strip_html_tags(text)
    text = remove_accented_chars(text)
    text = expand_contractions(text, CONTRACTION_MAP)
    text = lowercase_text(text)
    text = remove_special_characters(text)  # apply special character removal without removing digits

    # Convert the text to a doc type to process and parse through tokens
    doc = nlp(text)
    doc = add_measurement_entities(doc)

    # Process and filter out numbers that are not part of dates or measurements
    doc = process_and_filter_numbers(doc)

    # Construct the final string from processed tokens
    processed_text = ' '.join([token.text for token in doc])
    return processed_text

def preprocess_dataframe(df, column):
    """
    Applies the preprocess_text function to a specific column in a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the text data.
        column_name (str): The name of the column to preprocess.

    Returns:
        pandas.DataFrame: DataFrame with the text column preprocessed.
    """
    # Apply the preprocess_text function to the specified column
    df[column] = df[column].apply(preprocess_text)
    return df