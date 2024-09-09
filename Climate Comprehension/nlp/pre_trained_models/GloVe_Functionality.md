<h1 style="text-align: center;">Global Vectors for Word Representation (GloVe): 100 Dimensional Embeddings</h1>

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for generating word embeddings, developed by researchers at Stanford University. Word embeddings are dense vector representations of words that capture their meanings, semantic relationships, and syntactic properties. The "100D" in GloVe 100D indicates that each word is represented as a 100-dimensional vector.

### Co-occurrence Matrix

The GloVe model begins by constructing a co-occurrence matrix \( X \), where each entry \( X_{ij} \) represents the number of times word \( j \) appears in the context of word \( i \). The context is typically defined by a fixed window size around each word in the corpus.

### Objective Function

The objective of GloVe is to find word vectors \( w \) and context word vectors \( \tilde{w} \) such that their dot product approximates the logarithm of the words' probability of co-occurrence. The objective function to minimize is:

\[ J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij}) \right)^2 \]

Where:
- \( V \) is the size of the vocabulary.
- \( w_i \) and \( \tilde{w}_j \) are the word vectors for the target and context words, respectively.
- \( b_i \) and \( \tilde{b}_j \) are bias terms.
- \( f(X_{ij}) \) is a weighting function that assigns lower weight to rare co-occurrences.

### Weighting Function

The weighting function \( f(X_{ij}) \) is defined as:

\[ f(X_{ij}) = \begin{cases} 
\left( \frac{X_{ij}}{X_{\text{max}}} \right)^\alpha & \text{if } X_{ij} < X_{\text{max}} \\
1 & \text{otherwise}
\end{cases} \]

Typically, \( \alpha = 0.75 \) and \( X_{\text{max}} \) is a tunable parameter.

### Semantic Relationships

One of the key strengths of GloVe embeddings is their ability to capture semantic relationships between words. For example, in the GloVe 100D space:

- The vector difference between "king" and "queen" is similar to the vector difference between "man" and "woman":
  
  \[
  \text{king} - \text{man} \approx \text{queen} - \text{woman}
  \]
  
  This can be represented as:
  
  \[
  \text{vec}(\text{king}) - \text{vec}(\text{man}) \approx \text{vec}(\text{queen}) - \text{vec}(\text{woman})
  \]

### Analogies

GloVe embeddings can also be used to solve word analogies. For instance, the analogy "man is to king as woman is to queen" can be found by performing vector arithmetic:

\[
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})
\]

In this way, GloVe embeddings enable models to understand and process analogical relationships between words.

### Clustering and Similarity

Words with similar meanings are represented by vectors that are close to each other in the embedding space. For example:

- Synonyms: The vectors for "happy" and "joyful" will be close to each other.
- Antonyms: The vectors for "happy" and "sad" will also show a meaningful relationship but will be positioned differently compared to synonyms.

Mathematically, the similarity between two word vectors can be measured using the cosine similarity:

\[
\text{cosine\_similarity}(w_i, w_j) = \frac{w_i \cdot w_j}{\|w_i\| \|w_j\|}
\]

### Contextual Similarity

GloVe embeddings capture contextual similarity by ensuring that words appearing in similar contexts have similar vectors. For example:

- Words like "cat" and "dog" will have similar vectors because they often appear in similar contexts (e.g., "The [animal] is playing").
