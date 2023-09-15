import utilities
from sklearn.feature_extraction.text import CountVectorizer


def get_sentences(filename):
    sherlock_holmes_text = utilities.read_text_file(filename)
    sherlock_holmes_text = utilities.preprocess_text(sherlock_holmes_text)
    sentences = utilities.divide_into_sentences_nltk(sherlock_holmes_text)
    return sentences


def get_new_sentence_vector(sentence, vectorizer):
    new_sentence_vector = vectorizer.transform([sentence])
    return new_sentence_vector


def show_matrix(vectorizer, X):
    # Convert sparse matrix to COO format
    X_coo = X.tocoo()
    feature_names = vectorizer.get_feature_names_out()
    # for idx, feature in enumerate(feature_names):
    #     print(f"Index: {idx}, Feature: {feature}")
    # Iterate through the COO matrix
    denseX = X.todense()
    print(f"Dense Matrix: {denseX}")
    print(f"Sparse Matrix: \n")
    for row, col, value in zip(X_coo.row, X_coo.col, X_coo.data):
        print(
            f"Row: {row}, Index: {col}, Value: {value} Term: {feature_names[col]}")

#  takes in a list of sentences and returns the vectorizer
#  object and the final matrix representation of the sentences. We will use the vectorizer
#  object later on to encode new, unseen sentences.


def create_vectorizer(sentences):
    # df = document frequency, In this case, the vectorizer will consider words that appear in less than 80% of all documents.
    vectorizer = CountVectorizer(max_df=0.8, stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return (vectorizer, X)


def main():
    sentences = get_sentences(
        "./data/Gutenberg/Chapter01/sherlock_holmes_1.txt")  # returns a list of sentences
    i = 0
    for sentence in sentences:
        print(f"{i} : {sentence}")
        i = i + 1
    (vectorizer, X) = create_vectorizer(sentences)
    #print(vectorizer, X)
    show_matrix(vectorizer, X)
    new_sentence = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
    new_sentence_vector = get_new_sentence_vector(new_sentence, vectorizer)
    #print(f"New Sentence {new_sentence_vector}")
    print("-------------------------")
    show_matrix(vectorizer, new_sentence_vector)
    analyze = vectorizer.build_analyzer()
    print(analyze(new_sentence))


if (__name__ == "__main__"):
    main()
