#import gensim
from gensim.models import KeyedVectors
import numpy as np
import utilities

w2vec_model_path = "./data/NLPL/model.bin"


def main():
    model = utilities.load_w2vec_model(w2vec_model_path)
    print(model['holmes'])
    print("Matching\n")
    print(model.most_similar(['holmes'], topn=15))
    sentence = "It was not that he felt any emotion akin to love for Irene Adler."
    word_vectors = utilities.get_w2vec_word_vectors(sentence, model)
    sentence_vector = utilities.get_w2vec_sentence_vector(word_vectors)
    words = ['banana', 'apple', 'computer', 'strawberry']
    print(f"Non matching word for {words}\n")
    print(model.doesnt_match(words))
    print("Most similar to the word - cup\n")
    word = "cup"
    words = ['glass', 'computer', 'pencil', 'watch']
    print(model.most_similar_to_given(word, words))


if (__name__ == "__main__"):
    main()
