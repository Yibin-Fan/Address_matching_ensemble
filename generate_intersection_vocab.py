import gensim
import numpy as np


def generate_intersection_vocab(glove_vectors_file, word2vec_model_file, output_file):
    # Load Word2Vec model
    w2v_model = gensim.models.Word2Vec.load(word2vec_model_file)
    w2v_vocab = set(w2v_model.wv.index_to_key)

    # Load GloVe vectors
    glove_vocab = set()
    with open(glove_vectors_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split()[0]
            glove_vocab.add(word)

    # Find intersection
    intersection_vocab = glove_vocab.intersection(w2v_vocab)

    # Save intersection vocabulary
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted(intersection_vocab):
            f.write(f"{word}\n")

    return len(intersection_vocab)


vocab_size = generate_intersection_vocab('GloVe/vectors.txt', 'model/word2vec.model', 'data/vocab/intersection_vocab.txt')
print(f"Intersection vocabulary size: {vocab_size}")