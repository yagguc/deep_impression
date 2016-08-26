import auditory_stream
import chainer
import visual_stream

### MODEL ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__(
            aud = auditory_stream.ResNet18(),
            vis = visual_stream.ResNet18(),
            fc = chainer.links.Linear(512, 5, initialW = chainer.initializers.HeNormal())
        )
    
    def __call__(self, x):
        h = [self.aud(True, chainer.Variable(chainer.cuda.to_gpu(x[0]), True)), chainer.functions.expand_dims(chainer.functions.sum(self.vis(True, chainer.Variable(chainer.cuda.to_gpu(x[1][:256]), True)), 0), 0)]
        
        for i in xrange(256, x[1].shape[0], 256):
            h[1] += chainer.functions.expand_dims(chainer.functions.sum(self.vis(True, chainer.Variable(chainer.cuda.to_gpu(x[1][i : i + 256]), True)), 0), 0)
        
        h[1] /= x[1].shape[0]
        
        return chainer.cuda.to_cpu(((chainer.functions.tanh(self.fc(chainer.functions.concat(h))) + 1) / 2).data[0])
### MODEL ###
