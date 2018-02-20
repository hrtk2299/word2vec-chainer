# -*- coding: utf-8 -*-
import argparse

import chainer

from continuous_bow import ContinuousBoW
from utility import load_wordidmap

n_result = 5


def main():
    parser = argparse.ArgumentParser('Display words similar to input words.')
    parser.add_argument('word2vec_model', type=str, help='word2vec model file in npz format')
    parser.add_argument('wordidmap_filepath', type=str, help='A word-ID map filepath.')

    args = parser.parse_args()

    model_filepath = args.word2vec_model
    index2word, _ = load_wordidmap(args.wordidmap_filepath)
    n_vocab = len(index2word)
    word2index = {word: idx for idx, word, in index2word.items()}
    word2vec = ContinuousBoW(n_vocab, 100)
    chainer.serializers.load_npz(model_filepath, word2vec)

    word_vectors = word2vec.embed.W.data

    while True:
        s = input('>> ')
        if s not in word2index:
            print(f'{s} is not found')
            continue
        v = word_vectors[word2index[s]]
        similarity = word_vectors.dot(v)
        print(f'query: {s}')
        for n, idx in enumerate((-similarity).argsort()[:n_result], start=1):
            print(f'{n}: {index2word[idx]} ({similarity[idx]})')


if __name__ == '__main__':
    main()
