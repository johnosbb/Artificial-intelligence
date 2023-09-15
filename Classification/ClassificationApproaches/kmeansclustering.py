import utilities
import pandas as pd
import pickle
from sklearn.cluster import KMeans


bbc_dataset = "./data/BBC/bbc-text.csv"
stopwords_file_path = "./data/Stopwords/stopwords.csv"

# K-Means clustering is a popular unsupervised machine learning algorithm used for clustering data points into K distinct groups
# or (clusters) based on their similarities.
# The "K" in K-Means refers to the number of clusters that the algorithm aims to find.


# The basic principle of K-Means clustering can be summarized in the following steps:
# Initialization: First, K initial cluster centroids are randomly chosen from the data points or initialized using some other strategy (e.g., k-means++). These centroids represent the center of each cluster.
# Assignment: Each data point is assigned to the nearest centroid based on some distance metric (usually Euclidean distance). This step is also known as the assignment step.
# Update: The cluster centroids are updated by calculating the mean (average) of all the data points assigned to each cluster. This step is also known as the update step.
# Iteration: Steps 2 and 3 are repeated iteratively until the centroids no longer change significantly, or a specified number of iterations are reached.
# Convergence: The algorithm converges when the centroids stabilize, and the data points are assigned to their final clusters.
# The objective of K-Means is to minimize the within-cluster sum of squares (WCSS), which measures the total distance of each data point from its assigned centroid within the cluster. The algorithm iteratively tries to find the optimal centroids that minimize the WCSS.
# K-Means is a fast and scalable algorithm, making it suitable for large datasets. However, it requires specifying the number of clusters K beforehand, which can be a limitation when the optimal value of K is unknown. In practice, the choice of K is often determined using techniques like the elbow method or silhouette analysis.
# It is important to note that K-Means can converge to a local minimum, meaning the final clusters depend on the initial centroid positions. To mitigate this issue, multiple runs with different initializations can be performed, and the best result is selected based on the WCSS or other evaluation metrics.


def print_report(predicted_data):
    for topic in predicted_data.keys():
        print(topic)
        for prediction in predicted_data[topic].keys():
            print("Cluster number: ", prediction, "number of items: ",
                  len(predicted_data[topic][prediction]))


def print_most_common_words_by_cluster(all_training, km, num_clusters, stopwords):
    clusters = km.labels_.tolist()
    docs = {'text': all_training, 'cluster': clusters}
    frame = pd.DataFrame(docs, index=[clusters])
    for cluster in range(0, num_clusters):
        this_cluster_text = frame[frame['cluster'] == cluster]
        all_text = " ".join(this_cluster_text['text'].astype(str))
        top_200 = utilities.get_most_frequent_words(all_text, stopwords, 200)
        print(cluster)
        print(top_200)
    return frame


def main():
    number_of_clusters = 5
    stopwords = utilities.get_stemmed_stopwords_from_cvs(stopwords_file_path)
    data_dict = utilities.get_cvs_data_as_dictionary(bbc_dataset)
    (train_dict, test_dict) = utilities.divide_data_using_dictionary_keys(data_dict)
    all_training = []
    all_test = []
    for topic in train_dict.keys():
        # all_training will contain the articles but not the topic keys
        all_training = all_training + train_dict[topic]
    for topic in test_dict.keys():
        # all_test will contain the articles but not the topic keys
        all_test = all_test + test_dict[topic]
    vectorizer, matrix = utilities.create_tfid_vectorizer_and_matrix(
        all_training, stopwords)
    # n_clusters=5: This parameter specifies the number of clusters (groups) the algorithm should attempt to find. In this case, the model is configured to find 5 clusters.
    # returns a K Means Clustering to which we can pass a Matrix
    km = KMeans(n_clusters=number_of_clusters,
                init='k-means++', random_state=0)
    # km.fit: this is producing an error config = get_config().split() AttributeError: 'NoneType' object has no attribute 'split'
    # In his book The Hundred Page Machine Learning Book, Andriy Burkov describes an algorithm to determine the number of clusters most likely present in a model.
    km.fit(matrix)
    predicted_data = utilities.make_predictions(test_dict, vectorizer, km)
    print_report(predicted_data)
    print_most_common_words_by_cluster(
        all_training, km, number_of_clusters, stopwords)
    # After running this line of code, you will have the KMeans model saved in the file "bbc_kmeans.pkl," which you can later load and use for predictions or analysis without retraining the model from scratch.
    pickle.dump(km, open("./data/BBC/bbc_kmeans.pkl", "wb"))


if (__name__ == "__main__"):
    main()
