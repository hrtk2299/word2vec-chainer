# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I


class ContinuousBoW(chainer.Chain):
    def __init__(self, n_vocab, n_units):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.out = L.Linear(n_units, n_vocab, initialW=0)

    def __call__(self, x):
        e = self.embed(x)
        h = F.sum(e, axis=1) * (1. / x.shape[1])
        h = self.out(h)
        return h
