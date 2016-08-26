import audiovisual_stream
import chainer.serializers
import librosa
import numpy
import skvideo.io

def load_audio(data):
    return librosa.load(data, 16000)[0][None, None, None, :]

def load_model():
    model = audiovisual_stream.ResNet18().to_gpu()
    
    chainer.serializers.load_npz('./model', model)
    
    return model

def load_video(data):
    videoCapture = skvideo.io.VideoCapture(data, (456, 256))
    
    videoCapture.open()
    
    x = []
    
    while True:
        retval, image = videoCapture.read()
        
        if retval:
            x.append(numpy.rollaxis(image, 2))
        else:
            break
    
    return numpy.array(x, 'float32')

def predict_trait(data, model):
    x = [load_audio(data), load_video(data)]
    
    return model(x)
