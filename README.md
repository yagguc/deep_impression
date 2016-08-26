# deep_impression
Deep Impression: Audiovisual Deep Residual Networks for Multimodal Apparent Personality Trait Recognition

This repository contains the implementation of the model that won the third place in the [ChaLearn First Impressions Challenge @ ECCV2016][1].

##### The details of the model can be found in:

Yağmur Güçlütürk, Umut Güçlü, Marcel van Gerven, Rob van Lier. Deep Impression: Audiovisual Deep Residual Networks for Multimodal Apparent Personality Trait Recognition. ChaLearn Looking at People Workshop on Apparent Personality Analysis, ECCV Workshop proceedings, LNCS, Springer, 2016, in press.

######  ABSTRACT
Here, we develop an audiovisual deep residual network for multimodal apparent personality trait recognition. The network is trained end-to-end for predicting the Big Five personality traits of people from their videos. That is, the network does not require any feature engineering or visual analysis such as face detection, face landmark alignment or facial expression recognition. Recently, the network won the third place in the [ChaLearn First Impressions Challenge @ ECCV2016][1] with a test accuracy of 0.9109.

[1]: http://gesture.chalearn.org/2016-looking-at-people-eccv-workshop-challenge

##### The demo of the model can be found in:

[demo.ipynb][2].

[2]: https://github.com/yagguc/deep_impression/blob/master/demo.ipynb

###### REQUIREMENTS
System: CUDA Toolkit, cuDNN (and a suitable NVIDIA GPU).  
Python: [chainer][3], [librosa][4], [numpy][5], [skvideo][6].

[3]: http://docs.chainer.org/en/stable/install.html
[4]: http://librosa.github.io/librosa/install.html
[5]: http://scipy.org/install.html
[6]: http://www.scikit-video.org/stable/

*Please cite the above paper if you use the model in your work.*
