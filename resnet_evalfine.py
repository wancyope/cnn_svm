import chainer
import chainer.functions as F
from chainer import links as L

class Encoder(chainer.Chain):

    def __init__(self):
        super(Encoder, self).__init__(
            model = L.ResNet50Layers()
        )
    def __call__(self, x):
        h = self.model(x,layers=['res5'])
        h = h['res5']
        h = F.average_pooling_2d(h, ksize=(12,8), stride=1)
        return h
