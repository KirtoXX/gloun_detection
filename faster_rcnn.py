import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from gluoncv.model_zoo.faster_rcnn.faster_rcnn import get_faster_rcnn
from gluoncv.model_zoo.faster_rcnn.resnet50_v2a import resnet50_v2a

def Faster_rcnn(pretrained=False, pretrained_base=True,**kwargs):
    #--------init class-----
    classes = []
    for i in range(60):
        classes.append(str(i + 1))

    #----------pretrained------------
    pretrained_base = False if pretrained else pretrained_base

    #----------backbone net----------
    base_network = resnet50_v2a(pretrained=pretrained_base)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['rescale'] + ['layer' + str(i) for i in range(4)]:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))

    #-----------train_parm------------
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*stage(2|3|4)_conv'])
    model = get_faster_rcnn('resnet50_v2a', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns,
                           pretrained=pretrained, **kwargs)
    return model
