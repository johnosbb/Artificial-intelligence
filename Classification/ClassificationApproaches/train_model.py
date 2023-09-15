import utilities
import pickle
import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors

word2vec_model_path = "./data/word2vec.model"
books_dir = "./data/Gutenberg/textfiles"
evaluation_file = "./data/questions-words.txt"
pretrained_model_path = "./data/NLPL/model.bin"


def main():
    sentences = utilities.get_all_book_sentences(books_dir)
    sentences = [utilities.tokenize_nltk(s.lower()) for s in sentences]
    model = utilities.train_word2vec(sentences, word2vec_model_path)
    w1 = "river"
    words = model.wv.most_similar(positive=[w1], topn=10)
    print(words)
    # test_model()
    model = pickle.load(open(word2vec_model_path, 'rb'))
    (analogy_score, analogy_list) = utilities.evaluate_model(model, evaluation_file)
    print(analogy_list)
    print(analogy_score)

    pretrained_model = KeyedVectors.load_word2vec_format(
        pretrained_model_path, binary=True)  # load the pretrained model
    (analogy_score, analogy_list) = pretrained_model.evaluate_word_analogies(
        evaluation_file)
    print("Pre-trained model\n")
    print(analogy_list)
    print(analogy_score)
    # The pretrained model was trained on a much larger corpus, and, predictably, performs
    # better. However, it still doesn't get a very high score. Your evaluation should be based on
    # the type of text you are going to be working with, since the file that's provided with the
    # gensim package is a generic evaluation

    # (analogy_score, word_list) = model.wv.evaluate_word_analogies(
    #     datapath('questions-words.txt'))
    # print(analogy_score)
    # pretrained_model = KeyedVectors.load_word2vec_format(
    #     pretrained_model_path, binary=True)
    # (analogy_score, word_list) = pretrained_model.evaluate_word_analogies(
    #     datapath('questions-words.txt'))
    # print(analogy_score)


if (__name__ == "__main__"):
    main()
