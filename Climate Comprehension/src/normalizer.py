"""
------------------------------------------------------------------------------------------------------------
                                Advanced NLP Preprocessor
------------------------------------------------------------------------------------------------------------

By: Elijah Taber
"""

import pandas as pd
import re
from typing import Dict
from bs4 import BeautifulSoup
import unicodedata
import spacy
import nltk
from nltk.corpus import wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from src.contractions import CONTRACTION_MAP # dictionary of common contractions

class TextNormalizer:
    """
    A class designed to perform advanced text normalization for NLP tasks such as text classification,
    sentiment analysis, and more. This class enhances text preprocessing by applying a series of
    specific normalization operations, each aimed at cleaning and standardizing text data to improve the
    effectiveness of NLP models. This is considered step 1 in the NLP preprocessing pipeline and does
    include other techniques such as tokenization, vectorization, and numerical encoding. This is 
    designed to clean and standardize the text data before further preprocessing steps.

    Attributes:
        nlp (spacy.lang): Loaded spaCy language model configured without parser and ner for efficient text processing.
        tokenizer (ToktokTokenizer): Tokenizer used for breaking text into tokens.
        stopword_list (List[str]): List of English stopwords from NLTK to remove non-informative words from the text.
        normalization_steps (dict): Dictionary mapping text normalization functions to their respective flag,
                                    controlling which operations are applied during text normalization.
    """
    
    def __init__(self):
        """
        Initializes the TextNormalizer instance with various components used in text preprocessing.

        Attributes:
            nlp (spacy.lang): SpaCy language model for English, specifically the small model 'en_core_web_sm'.
                            The parser and named entity recognizer (NER) components are disabled to
                            speed up processing for tasks that don't need full syntactic or entity analysis.
            nlp.max_length (int): The maximum length of text that the SpaCy model can process. Set to 10 million
                                characters, which can be adjusted depending on the expected length of input texts.
            tokenizer (ToktokTokenizer): Tokenizer used to split text into tokens. The Toktok tokenizer is a fast,
                                        rule-based tokenizer which is used as an alternative to SpaCy's built-in tokenizer.
            stopword_list (list): A list of stopwords retrieved from the NLTK library, specifically for the English language.
                                These stopwords are common words that are usually removed during text preprocessing to focus
                                on more meaningful words.
            normalization_steps (dict): A dictionary (hash map) mapping preprocessing step names to their corresponding 
                                        functions or lambda expressions.

        Each step in `normalization_steps` is designed to be applied sequentially to a text string to perform comprehensive
        text normalization and cleaning. Users can modify or extend these steps according to their specific text processing needs.
        """
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.nlp.max_length = 50_000_000  # adjustable based on expected text length limits
        self.tokenizer = ToktokTokenizer()
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.normalization_steps = {
            'html_stripping': self.strip_html_tags,
            'url_and_email_removal': self.remove_urls_and_emails,
            'accented_char_removal': self.remove_accented_chars,
            'contraction_expansion': lambda text: self.expand_contractions(text, CONTRACTION_MAP),
            'text_lower_case': lambda x: x.lower(),
            'special_char_removal': lambda x: self.remove_special_characters(x, True),
            'stopword_removal': lambda x: self.remove_stopwords(x, True),
            'repeated_char_removal': self.remove_repeated_characters,
            'text_lemmatization': self.lemmatize_text
        }

    def strip_html_tags(self, text: str) -> str:
        """
        Strips HTML tags and all embedded JavaScript/CSS content from the text, using BeautifulSoup to parse HTML content.

        Parameters:
            text (str): The text string from which HTML and script tags will be removed.

        Returns:
            str: A cleaned text string without HTML tags and script/style content. All consecutive whitespace characters
                 are replaced by a single space, and leading/trailing whitespace is removed.
        """
        soup = BeautifulSoup(text, "html.parser")
        for script_or_style in soup(["script", "style", "iframe"]):  # removing script, style, and iframe elements
            script_or_style.decompose()
        cleaned_text = soup.get_text(separator=' ', strip=True)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # normalize whitespace to a single space where needed
        return cleaned_text

    def remove_accented_chars(self, text: str) -> str:
        """
        Transforms accented characters in a string into their unaccented equivalent, improving the uniformity of text.

        Parameters:
            text (str): Text from which accented characters will be normalized to ASCII characters.

        Returns:
            str: The modified text with accented characters replaced by their ASCII equivalents. This helps in reducing
                 variation caused by different forms of the same letter due to accent marks.
        """
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    @staticmethod
    def expand_contractions(text: str, contraction_mapping: Dict[str, str]) -> str:
        """
        Expands contractions found in the text using a mapping dictionary with common contractions.

        Parameters:
            text (str): Text containing contractions to be expanded.
            contraction_mapping (Dict[str, str]): A dictionary where keys are contractions and values are the expanded form.

        Returns:
            str: Text with contractions expanded.
        """
        # Create a regular expression pattern that matches contractions in the text using the keys of the contraction mapping
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0) # get the matched contraction
            first_char = match[0] # get the first character of the contraction
            expanded_contraction = contraction_mapping.get(match.lower(), match) # get the expanded form from the mapping
            expanded_contraction = first_char + expanded_contraction[1:] # capitalize the first character if needed
            return expanded_contraction
        expanded_text = contractions_pattern.sub(expand_match, text) # sub the matched contraction with the expanded form
        expanded_text = re.sub("'", "", expanded_text)  # remove any remaining single quotes
        return expanded_text

    def remove_special_characters(self, text: str, remove_digits: bool = False) -> str:
        """
        Removes special characters from the text, preserving only alphanumeric characters and whitespace.
        This method can also remove digits if specified, making it useful for text-only analysis.

        Parameters:
            text (str): The text from which to remove special characters.
            remove_digits (bool): A flag to determine whether digits should also be removed. Defaults to False.

        Returns:
            str: The cleaned text with special characters (and optionally digits) removed. This processing is essential
                 for tasks that require purely alphabetical input.
        """
        pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
        text = re.sub(pattern, '', text)
        return text

    def remove_repeated_characters(self, text: str) -> str:
        """
        Reduces repetition in characters within tokens to improve the quality of tokenization. It keeps the word if
        recognized by WordNet, ensuring that meaningful words aren't incorrectly shortened.

        Parameters:
            text (str): The text from which repeated characters will be reduced.

        Returns:
            str: The text with repeated characters reduced.
        """
        repeat_pattern = re.compile(r'(\w*)(\w)\2{2,}(\w*)')
        match_substitution = r'\1\2\3'

        def replace(old_word: str) -> str:
            if wordnet.synsets(old_word):
                return old_word
            new_word = repeat_pattern.sub(match_substitution, old_word)
            return replace(new_word) if new_word != old_word else new_word

        tokens = self.tokenizer.tokenize(text)
        corrected_tokens = [replace(word) for word in tokens]
        return ' '.join(corrected_tokens)

    def lemmatize_text(self, text: str) -> str:
        """
        Converts words in the text to their dictionary form (lemma) using spaCy's NLP model.
        Unlike stemming, lemmatization considers the context and converts the word to its
        meaningful base form, which is better for tasks that depend on accurate linguistic representations.
        Pronouns are kept in their original form to maintain their grammatical integrity.
        
        Parameters:
            text (str): The text to be lemmatized.
        
        Returns:
            str: The text after applying lemmatization, where each word is converted to its lemma,
                with pronouns retained in their original form.
        """
        doc = self.nlp(text)
        lemmatized_tokens = []
        for token in doc:
            if token.lemma_ == '-PRON-' or token.tag_ in ['PRP', 'PRP$']:
                lemmatized_tokens.append(token.text)
            else:
                lemmatized_tokens.append(token.lemma_)
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text

    def remove_stopwords(self, text: str, is_lower_case: bool = False) -> str:
        """
        Removes common stopwords from the text, which are typically words that add little value to text understanding
        in NLP tasks like classification.

        Parameters:
            text (str): The text from which stopwords are to be removed.
            is_lower_case (bool): Specifies whether the text has been converted to lower case before this operation.
                                  This helps in accurately matching words to the stopwords list.

        Returns:
            str: The text with stopwords removed, which may help in focusing on more meaningful words in NLP analysis.
        """
        tokens = self.tokenizer.tokenize(text)
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_urls_and_emails(self, text: str) -> str:
        """
        Removes URLs and email addresses from the text using basic string operations.

        Parameters:
            text (str): The text from which URLs and email addresses are to be removed.

        Returns:
            str: The cleaned text with URLs and email addresses removed.
        """
        words = text.split()
        clean_words = [word for word in words if not (word.startswith("http") or "@" in word or ".com" in word)]
        clean_text = " ".join(clean_words)
        return clean_text

    def normalize_text(self, text: str) -> str:
        """
        Applies predefined text normalization techniques to clean and prepare text for further NLP tasks.
        
        Parameters:
            text (str): The original text to be normalized.
        
        Returns:
            str: The normalized text ready for NLP tasks.
        """
        for step, func in self.normalization_steps.items():
            text = func(text)  # apply each function in the normalization_steps
        return text.strip()  # clean up any leading/trailing whitespace

    def normalize_corpus(self, corpus: pd.Series) -> pd.Series:
        """
        Normalizes a pandas Series of text corpuses by applying the defined text normalization processes.

        Parameters:
            corpus (pd.Series): A pandas Series containing text corpuses to be normalized.

        Returns:
            pd.Series: A pandas Series of normalized text corpuses.
        """
        return corpus.apply(self.normalize_text)

    def normalize_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Normalizes a specific text column in a dataframe by applying the full text normalization pipline.

        Parameters:
            df (pd.DataFrame): The dataframe containing the text column to be normalized.
            text_column (str): The name of the column containing text to be normalized.

        Returns:
            pd.DataFrame: The dataframe with the normalized text column.
        """
        df[text_column] = self.normalize_corpus(df[text_column])
        return df