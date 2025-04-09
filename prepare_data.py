import jieba
import json


def load_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary

def tokenize_and_index(text, dictionary):
    words = jieba.lcut(text)
    indexed_words = []
    for word in words:
        if word != '':
            if word not in dictionary:
                dictionary[word] = len(indexed_words) + 1
            indexed_words.append(dictionary[word])
    return indexed_words

dictionary = load_dict('data/dict/word_dict.json')

# 读取数据
input_dir = ['train', 'test', 'valid']

for input in input_dir:
    input_file = 'data/dataset/' + input + '/address.txt'
    output_file1 = 'data/dataset/' + input + '/addr1_tokenized.txt'
    output_file2 = 'data/dataset/' + input + '/addr2_tokenized.txt'
    output_file3 = 'data/dataset/' + input + '/labels.txt'
    with open(input_file, 'r', encoding='utf-8') as f:
        addr1 = []
        addr2 = []
        labels = []
        for line in f.readlines():
            # 拆分数据
            columns = line.strip().split('\t')

            if len(columns) == 3:
                addr1.append(columns[0])
                addr2.append(columns[1])
                labels.append(columns[2])

    with open ('data/dataset/' + input + '/addr1_tokenized.txt', 'w', encoding='utf-8') as f1, \
         open('data/dataset/' + input + '/addr2_tokenized.txt', 'w', encoding='utf-8') as f2, \
         open('data/dataset/' + input + '/labels.txt', 'w', encoding='utf-8') as f3:
        for i in range(len(addr1)):
            addr1_tokens = tokenize_and_index(addr1[i], dictionary)
            addr2_tokens = tokenize_and_index(addr2[i], dictionary)

            f1.write(' '.join(map(str, addr1_tokens)) + '\n')
            f2.write(' '.join(map(str, addr2_tokens)) + '\n')
            f3.write(labels[i] + '\n')