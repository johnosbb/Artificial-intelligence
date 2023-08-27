# Text Classification in NLP


## The Text Classification Process

![Text Classification Process](https://github.com/johnosbb/Artificial-intelligence/blob/main/ProcessFlow_TextClassification.png)

## Glossary of NLP Terms

- __Bag_of Words__: A "Bag of Words" (BoW) is a simple and fundamental technique used for text analysis and feature extraction. It's a way to represent text data, such as sentences or documents, as numerical vectors that can be used in machine learning algorithms.
- __Dimensionality Expansion__: This process is also called "vector space transformation." It aims to enrich the original bag of words representations to capture more complex relationships between words or documents. These representations of vectors into higher-dimensional spaces using techniques like TF-IDF, word embeddings, LSA, or neural networks allow one to capture more nuanced and meaningful information from text data.
- __Deep Learning__: Deep learning, in the context of Natural Language Processing (NLP), refers to a subset of machine learning techniques that involve the use of deep neural networks to understand, process, and generate human language. The term "deep" in deep learning refers to the presence of multiple hidden layers in a neural network. Deep learning models used in NLP often have many layers, which allow them to capture intricate patterns and representations in text data.
- __Vectorizer__:  A component or tool that is used to convert raw text data into numerical vectors.
- __Vectors__: A vector is a one-dimensional array of numbers or values. These are typically used to represent individual data points or features and can represent quantities like position, direction, or a list of values. A vector is often represented as a column matrix (n x 1) or a row matrix (1 x n), where 'n' is the number of elements in the vector.
- __Matrix__: A matrix is a two-dimensional array of numbers or values consisting of rows and columns and is often used to represent data in a tabular format.
- __Dense Matrix__: A dense matrix is one in which most of the elements are non-zero and explicitly stored. It is characterized by having a value assigned to almost every element, regardless of whether it's zero or non-zero. Dense matrices are memory-intensive because they store every element, even if many of them are zero.
- __Sparse Matrix__: A sparse matrix is one in which most of the elements are zero, and only the non-zero elements are explicitly stored. It is characterized by having very few non-zero elements relative to the total number of elements in the matrix. Sparse matrices are memory-efficient because they only store non-zero elements and their positions.
- __Supervised Learning__: In supervised learning, the algorithm is trained on a labeled dataset. This means that for each input data point, the correct output or target is provided. The algorithm's objective is to learn a mapping from inputs to outputs by finding patterns and relationships in the labeled data.
- __Word Vectors__: Word vectors, or word embeddings, are numerical representations of words in multidimensional space through matrices. The purpose of the word vector is to get a computer system to understand a word. Computers cannot understand text efficiently. They can, however, process numbers quickly and well. For this reason, it is important to convert a word into a number.
- __Unsupervised Learning__: In unsupervised learning, the algorithm is trained on an unlabeled dataset. The algorithm's goal is to discover hidden patterns, structures, or relationships within the data without any predefined guidance.


## Word Vectors

Word vectors are like word representations in a computer. They have a fixed number of properties (dimensions) that the computer learns on its own. It figures this out by looking at how often words appear in texts and how they appear next to other words. This helps the computer understand how words are similar in terms of meaning.  The actual meanings encoded in each dimension are not explicitly defined. Word vectors are learned through neural network models, and the model determines the relationships and patterns during training by processing a vast amount of text data. Researchers often analyze word vectors to understand the relationships they capture, but the precise meaning of each dimension can be complex and context-dependent.
Consider a word vector for the word "king" with 300 dimensions. Each of these 300 dimensions is a separate numerical value, and together, they make up the complete word vector for a word like "king." Think of these dimensions as individual axes in a multi-dimensional space. In a 2D space, you have two axes (x and y), and you can pinpoint a location on a plane by specifying coordinates on both axes. In the case of a word vector with 300 dimensions, you have 300 separate axes, and each axis has its own numerical value. The distances and relationships between words are determined by the positions of their word vectors in this high-dimensional space. Words that are similar in meaning will have word vectors that are closer in this multi-dimensional space across all 300 dimensions, not just one. The individual values in these dimensions collectively capture various aspects of the word's meaning and context.


## Supervised versus Unsupervised Learning

![Learning](https://github.com/johnosbb/Artificial-intelligence/blob/main/supervised_unsupervised.png)

Supervised learning and unsupervised learning are two fundamental categories of machine learning, and they differ primarily in the way they are trained and the nature of the tasks they are designed to solve. Here are the key differences between the two:

### Training Data 

__Supervised Learning__: In supervised learning, the algorithm is trained on a labeled dataset. This means that for each input data point, the correct output or target is provided. The algorithm's objective is to learn a mapping from inputs to outputs by finding patterns and relationships in the labeled data.

__Unsupervised Learning__: In unsupervised learning, the algorithm is trained on an unlabeled dataset. There are no explicit target values provided. Instead, the algorithm's goal is to discover hidden patterns, structures, or relationships within the data without any predefined guidance.

### Task:

__Supervised Learning__: Supervised learning is used for tasks where the goal is to make predictions or classify data into predefined categories. Common tasks include classification (e.g., spam detection, image classification) and regression (e.g., predicting prices, estimating a continuous value).

__Unsupervised Learning__: Unsupervised learning is used for tasks focused on discovering the inherent structure or organization within data. Common tasks include clustering (grouping similar data points together), dimensionality reduction (reducing the number of features while preserving important information), and anomaly detection (identifying data points that deviate significantly from the norm).

### Output:

__Supervised Learning__: The output of a supervised learning algorithm is a predictive model that can make accurate predictions or classifications on new, unseen data.

__Unsupervised Learning__: The output of an unsupervised learning algorithm typically consists of insights about the data, such as cluster assignments or reduced-dimensional representations. It doesn't produce explicit predictions.

### Evaluation:

__Supervised Learning__: Supervised learning algorithms are evaluated based on their ability to make accurate predictions or classifications. Common evaluation metrics include accuracy, precision, recall, F1-score, and mean squared error (for regression).

__Unsupervised Learning__: Unsupervised learning algorithms are evaluated differently depending on the specific task. For clustering, metrics like silhouette score or inertia can be used. For dimensionality reduction, the quality of the reduced representation is assessed. Evaluation can be more subjective and context-dependent compared to supervised learning.

### Examples:

__Supervised Learning Examples__: Spam email classification, image recognition, sentiment analysis, and predicting house prices.

__Unsupervised Learning Examples__: Customer segmentation, topic modeling, image compression, and anomaly detection.

  

## Vectorizers
A vectorizer, in the context of natural language processing (NLP) and machine learning, is a component or tool that is used to convert raw text data into numerical vectors. It's a crucial step in preparing text data for machine learning algorithms, as most machine learning models require numerical input data. Vectorization allows you to represent text in a format that can be processed and analyzed by these models.

There are different types of vectorizers commonly used in NLP:

### Count Vectorizer:

Count vectorization, often referred to as the "Bag of Words" model, represents text by counting the frequency (or presence) of each word in a document. Each document is represented as a vector where each element corresponds to the count (or presence) of a specific word. It results in a high-dimensional and sparse vector representation. In some cases, you might want to normalize the vectors by dividing each count by the total number of words in the document. This helps account for varying document lengths. This simple technique has some limitations, such as not preserving word order or considering word semantics. 

### TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency):

TF-IDF vectorization represents text by considering both the term frequency (how often a word appears in a document) and the inverse document frequency (how unique a word is across all documents). It assigns a weight to each term in a document, emphasizing important terms while reducing the impact of common words.
Like count vectorization, it results in high-dimensional and sparse vectors.

### Word Embedding Vectorizer:

Word embedding techniques like Word2Vec, GloVe, and FastText map words to continuous-valued vectors in a lower-dimensional space. These embeddings capture semantic relationships between words, making them more informative than simple word counts. Word embeddings are dense vectors, meaning they have fewer dimensions than the vocabulary size.

### Character-level Vectorizer:

Character-level vectorization represents text by encoding characters or character n-grams as numerical vectors. It is useful for capturing morphological and spelling patterns in text.

### Custom Vectorizers:

In some cases, custom vectorization techniques are developed to address specific NLP tasks or domain-specific requirements.

The choice of vectorizer depends on the specific NLP task, the nature of the text data, and the machine learning algorithm being used. Vectorizers are an essential part of text preprocessing and feature engineering in NLP pipelines, enabling the transformation of text data into a format suitable for training and applying machine learning models.


## Classification Algorithms

### SVMs

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks, with a primary focus on classification. It's a powerful and versatile algorithm that works by finding the optimal hyperplane (decision boundary) that best separates different classes in a dataset.

SVM can be used to classify documents into different categories or classes based on the content of the text. Text data needs to be converted into numerical features that can be used by the SVM algorithm. Common techniques for this include bag-of-words (BoW) and TF-IDF (Term Frequency-Inverse Document Frequency) to represent the text as a vector. Given a labeled dataset of text documents with their corresponding categories, SVM learns to find the hyperplane that best separates the different classes. The goal is to maximize the margin between the classes while minimizing classification errors.  Once trained, the SVM can classify new, unseen text documents into one of the predefined categories based on their feature representations and the learned hyperplane.

__Key characteristics and advantages of SVM in text classification include__

- Effective for High-Dimensional Data: Text data is often high-dimensional because of the large number of unique words in a corpus. SVM can handle high-dimensional data efficiently.
- Robustness to Overfitting: SVM is less prone to overfitting compared to some other machine learning algorithms, making it suitable for text classification tasks with limited training data.
- Flexibility: SVMs can be adapted for multi-class classification by using techniques like one-vs-all or one-vs-one.
- Tuneable: SVMs have parameters, such as the kernel type and regularization parameter (C), that can be tuned to improve performance on specific datasets.
- Interpretability: The decision boundary learned by SVM is often interpretable, which can be useful for understanding why certain text documents were classified into specific categories.
- Note that SVMs can be computationally intensive, especially with large text datasets. In such cases, more efficient algorithms like linear classifiers or deep learning models might be considered. 


### Text Similarity Metrics

#### The Levenshtein distance

The Levenshtein distance, also known as the edit distance, is a metric used to measure the similarity or dissimilarity between two strings or sequences. It quantifies the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform one string into another. In the context of similarity, a lower Levenshtein distance implies greater similarity between the two strings.

__Methodology__

- Insertion: Adding a character to one of the strings. - Example: "kitten" and "kittens" have a Levenshtein distance of 1 because you need to insert an 's' to make them the same.
- Deletion: Removing a character from one of the strings. - Example: "flaw" and "law" have a Levenshtein distance of 1 because you need to delete the 'f' from the first string to make them the same.
- Substitution: Replacing a character in one of the strings with another character. - Example: "cat" and "hat" have a Levenshtein distance of 1 because you need to substitute 'c' with 'h' to make them the same.
- The Levenshtein distance can be useful in various applications, including spell-checking, DNA sequence alignment, and natural language processing. It provides a way to quantify how different two strings are, which is often used to determine the similarity or dissimilarity between words or phrases in a text analysis context. The smaller the Levenshtein distance between two strings, the more similar they are considered to be. 

