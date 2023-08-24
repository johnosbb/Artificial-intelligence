# Text Classification in NLP


## The Text Classification Process

![Text Classification Process](https://github.com/johnosbb/Artificial-intelligence/blob/main/ProcessFlow_TextClassification.png)

## Glossary of NLP Terms

- __Vectorizer__:  A component or tool that is used to convert raw text data into numerical vectors.
- __Vectors__: A vector is a one-dimensional array of numbers or values. These are typically used to represent individual data points or features and can represent quantities like position, direction, or a list of values. A vector is often represented as a column matrix (n x 1) or a row matrix (1 x n), where 'n' is the number of elements in the vector.
- __Matrix__: A matrix is a two-dimensional array of numbers or values consisting of rows and columns and is often used to represent data in a tabular format.
- __Dense Matrix__: A dense matrix is one in which most of the elements are non-zero and explicitly stored. It is characterized by having a value assigned to almost every element, regardless of whether it's zero or non-zero. Dense matrices are memory-intensive because they store every element, even if many of them are zero.
- __Sparse Matrix__: A sparse matrix is one in which most of the elements are zero, and only the non-zero elements are explicitly stored. It is characterized by having very few non-zero elements relative to the total number of elements in the matrix. Sparse matrices are memory-efficient because they only store non-zero elements and their positions.
- Supervised Learning: In supervised learning, the algorithm is trained on a labeled dataset. This means that for each input data point, the correct output or target is provided. The algorithm's objective is to learn a mapping from inputs to outputs by finding patterns and relationships in the labeled data.


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

Count vectorization, often referred to as the "Bag of Words" model, represents text by counting the frequency of each word in a document.
Each document is represented as a vector where each element corresponds to the count of a specific word.
It results in a high-dimensional and sparse vector representation.

### TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency):

TF-IDF vectorization represents text by considering both the term frequency (how often a word appears in a document) and the inverse document frequency (how unique a word is across all documents).
It assigns a weight to each term in a document, emphasizing important terms while reducing the impact of common words.
Like count vectorization, it results in high-dimensional and sparse vectors.

### Word Embedding Vectorizer:

Word embedding techniques like Word2Vec, GloVe, and FastText map words to continuous-valued vectors in a lower-dimensional space.
These embeddings capture semantic relationships between words, making them more informative than simple word counts.
Word embeddings are dense vectors, meaning they have fewer dimensions than the vocabulary size.

### Character-level Vectorizer:

Character-level vectorization represents text by encoding characters or character n-grams as numerical vectors.
It is useful for capturing morphological and spelling patterns in text.

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



