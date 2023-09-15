import utilities
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
stopwords_file_path = "./data/Stopwords/stopwords.csv"


def main():
    sentences = utilities.get_sentences(
        "./data/Gutenberg/Chapter01/sherlock_holmes_1.txt")
    print(sentences)
    stopword_list = utilities.read_in_csv(stopwords_file_path)
    (vectorizer, matrix) = utilities.create_vectorizer_tfid(
        sentences, stopword_list)
    utilities.show_vector_matrix(vectorizer, matrix)
    analyze = vectorizer.build_analyzer()
    print(analyze("To Sherlock Holmes she is always _the_ woman."))
    print("------------------------------------------")
    (vectorizer, matrix) = utilities.create_char_vectorizer_tfid(sentences)
    utilities.show_vector_matrix(vectorizer, matrix)
    analyze = vectorizer.build_analyzer()
    print(analyze("To Sherlock Holmes she is always _the_ woman."))


if (__name__ == "__main__"):
    main()
