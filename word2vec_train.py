# -*- coding: utf-8 -*-
import argparse

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer import training
from chainer.training import extensions

from continuous_bow import ContinuousBoW
from utility import load_wordid_text


class WindowIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat
        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        contexts = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return contexts, center

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    center, contexts = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wordid_text_filepath', type=str, help='A word ID text filepath for learning.')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')

    args = parser.parse_args()

    batch_size = args.batchsize
    window = 5

    dataset = load_wordid_text(args.wordid_text_filepath)
    # dataset, word2index = create_id_dataset(f, end_symbol=".")

    train_iter = WindowIterator(dataset, window, batch_size, repeat=True)

    cbow = ContinuousBoW(max(dataset) + 1, 100)
    clf = L.Classifier(cbow, lossfun=F.softmax_cross_entropy)
    clf.compute_accuracy = False

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        clf.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(clf)

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    clf.to_cpu()
    chainer.serializers.save_npz('word2vec.npz', cbow)

    # with open('word2vec.model', 'w') as f:
    #     f.write('%d %d\n' % (len(index2word), args.unit))
    #     w = cuda.to_cpu(clf.predictor.embed.W.data)
    #     for i, wi in enumerate(w):
    #         v = ' '.join(map(str, wi))
    #         f.write('%s %s\n' % (index2word[i], v))


if __name__ == "__main__":
    main()
