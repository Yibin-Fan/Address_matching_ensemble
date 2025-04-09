import mittens
import gensim
import numpy as np
from collections import Counter


def load_corpus(token_file):
    sentences = []
    with open(token_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip().split())
    return sentences


def train_word2vec(token_file, vector_size, window_size, min_count, model_file):
    sentences = load_corpus(token_file)
    model = gensim.models.Word2Vec(sentences, vector_size=vector_size, window=window_size, min_count=min_count)
    model.save(model_file)


train_word2vec('data/token/tokenized_addresses.txt', 100, 5, 1, 'model/word2vec.model')