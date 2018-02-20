# -*- coding: utf-8 -*-
import argparse

import numpy as np
import chainer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from continuous_bow import ContinuousBoW
from utility import load_wordidmap


def main():
    parser = argparse.ArgumentParser('Visualization of distributed representation of words.')
    parser.add_argument('word2vec_model', type=str, help='word2vec model file in npz format')
    parser.add_argument('wordidmap_filepath', type=str, help='A word-ID map filepath.')

    n_plotlabel = 1000

    args = parser.parse_args()

    model_filepath = args.word2vec_model
    index2word, freq_id_list = load_wordidmap(args.wordidmap_filepath)
    n_vocab = len(index2word)
    sorted_id = np.argsort(freq_id_list)[::-1]

    word2vec = ContinuousBoW(n_vocab, 100)
    chainer.serializers.load_npz(model_filepath, word2vec)

    word_vectors = word2vec.embed.W.data[sorted_id]

    transformed_data = PCA(n_components=10).fit_transform(word_vectors)
    transformed_data = TSNE(n_components=2, perplexity=30).fit_transform(transformed_data)

    label = [index2word[i] for i in sorted_id[:n_plotlabel]]
    transformed_data = transformed_data[:n_plotlabel]

    plt.figure(figsize=(10, 8))

    for x, s in zip(transformed_data, label):
        plt.text(x[0], x[1], s, fontsize=8)

    xmin = transformed_data[:, 0].min() * 1.1
    xmax = transformed_data[:, 0].max() * 1.1
    ymin = transformed_data[:, 1].min() * 1.01
    ymax = transformed_data[:, 1].max() * 1.01

    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("component 0")
    plt.ylabel("component 1")
    plt.title("Visualization of distributed representation of words")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
