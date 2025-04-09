from collections import Counter

import jieba

def tokenize_address(corpus, token):
    with open(corpus, 'r', encoding='utf-8') as f:
        with open(token, 'w', encoding='utf-8') as t:
            for line in f:
                t.write(' '.join(jieba.cut(line)))
                t.write('\n')


def build_vocab(corpus, vocab_file, min_freq=1):
    counter = Counter()
    with open(corpus, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    with open(vocab_file, 'w', encoding='utf-8') as vf:
        for word, freq in counter.items():
            if freq >= min_freq:
                vf.write(f"{word}\t{freq}\n")

tokenize_address('data/corpus/shenzhen_corpus.txt', 'data/token/tokenized_addresses.txt')
build_vocab('data/token/tokenized_addresses.txt', 'data/vocab/vocab.txt', min_freq=1)