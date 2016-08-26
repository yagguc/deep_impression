import chainer

### BLOCK ###
class ConvolutionBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__(
            conv = chainer.links.Convolution2D(in_channels, out_channels, 7, 2, 3, initialW = chainer.initializers.HeNormal()),
            bn_conv = chainer.links.BatchNormalization(out_channels),
        )
    
    def __call__(self, TEST, x):
        h = self.conv(x)
        h = self.bn_conv(h, TEST)
        y = chainer.functions.relu(h)
        
        return y

class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__(
            res_branch2a = chainer.links.Convolution2D(in_channels, out_channels, 3, pad = 1, initialW = chainer.initializers.HeNormal()),
            bn_branch2a = chainer.links.BatchNormalization(out_channels),
            res_branch2b = chainer.links.Convolution2D(out_channels, out_channels, 3, pad = 1, initialW = chainer.initializers.HeNormal()),
            bn_branch2b = chainer.links.BatchNormalization(out_channels)
        )
    
    def __call__(self, TEST, x):
        h = self.res_branch2a(x)
        h = self.bn_branch2a(h, TEST)
        h = chainer.functions.relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h, TEST)
        h = x + h
        y = chainer.functions.relu(h)
        
        return y

class ResidualBlockA():
    def __init__(self):
        pass
    
    def __call__(self):
        pass

class ResidualBlockB(chainer.Chain):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockB, self).__init__(
            res_branch1 = chainer.links.Convolution2D(in_channels, out_channels, 1, 2, initialW = chainer.initializers.HeNormal()),
            bn_branch1 = chainer.links.BatchNormalization(out_channels),
            res_branch2a = chainer.links.Convolution2D(in_channels, out_channels, 3, 2, 1, initialW = chainer.initializers.HeNormal()),
            bn_branch2a = chainer.links.BatchNormalization(out_channels),
            res_branch2b = chainer.links.Convolution2D(out_channels, out_channels, 3, pad = 1, initialW = chainer.initializers.HeNormal()),
            bn_branch2b = chainer.links.BatchNormalization(out_channels)
        )
    
    def __call__(self, TEST, x):
        temp = self.res_branch1(x)
        temp = self.bn_branch1(temp, TEST)
        h = self.res_branch2a(x)
        h = self.bn_branch2a(h, TEST)
        h = chainer.functions.relu(h)
        h = self.res_branch2b(h)
        h = self.bn_branch2b(h, TEST)
        h = temp + h
        y = chainer.functions.relu(h)
        
        return y
### BLOCK ###

### MODEL ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__(
            conv1_relu = ConvolutionBlock(3, 32),
            res2a_relu = ResidualBlock(32, 32),
            res2b_relu = ResidualBlock(32, 32),
            res3a_relu = ResidualBlockB(32, 64),
            res3b_relu = ResidualBlock(64, 64),
            res4a_relu = ResidualBlockB(64, 128),
            res4b_relu = ResidualBlock(128, 128),
            res5a_relu = ResidualBlockB(128, 256),
            res5b_relu = ResidualBlock(256, 256)
        )
    
    def __call__(self, TEST, x):
        h = self.conv1_relu(TEST, x)
        h = chainer.functions.max_pooling_2d(h, 3, 2, 1)
        h = self.res2a_relu(TEST, h)
        h = self.res2b_relu(TEST, h)
        h = self.res3a_relu(TEST, h)
        h = self.res3b_relu(TEST, h)
        h = self.res4a_relu(TEST, h)
        h = self.res4b_relu(TEST, h)
        h = self.res5a_relu(TEST, h)
        h = self.res5b_relu(TEST, h)
        y = chainer.functions.average_pooling_2d(h, h.data.shape[2:])
        
        return y
### MODEL ### 
