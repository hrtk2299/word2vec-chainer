# -*- coding: utf-8 -*-


def create_id_dataset(text_iterator, end_symbol="."):
    id_dataset = []
    word2index = {"<bos>": 0, "<eos>": 1}
    for line in text_iterator:
        line = line.strip().strip(end_symbol)
        line += ' <eos>'
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
            id_dataset.append(word2index[word])

    return id_dataset, word2index


def load_wordid_text(wordid_text_filepath):
    wordid_list = []
    with open(wordid_text_filepath, 'r') as f:
        for line in f:
            line_split_list = (line.strip() + ',1').split(',')
            wordid_list += [int(x) for x in line_split_list]

    return wordid_list


def load_wordidmap(wordidmap_filepath):
    index2word = {}

    with open(wordidmap_filepath, 'r') as f:
        for line in f:
            idx_str, word = line.strip().split(',')
            index2word[int(idx_str)] = word

    return index2word
