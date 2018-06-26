from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
from gluoncv import model_zoo
from light_head_rcnn import My_LHRCNN
from gluoncv.model_zoo.faster_rcnn.faster_rcnn import faster_rcnn_resnet50_v2a_voc

def main():
    data = np.zeros((1,3,800,800))
    x = nd.array(data)
    #net = model_zoo.get_model('faster_rcnn_resnet50_v2a_voc',pretrained_base=True)
    net = faster_rcnn_resnet50_v2a_voc(pretrained_base=True)
    #net = My_LHRCNN()
    net.initialize()
    x1,x2,x3 = net(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    #net.initialize()


def main2():
    roi_data = np.zeros((100,5))
    roi_data = nd.array(roi_data)
    feat = np.zeros((1,10*3*3,300,300))
    feat = nd.array(feat)
    result = nd.contrib.PSROIPooling(feat,roi_data,spatial_scale=1/16,output_dim=10,pooled_size=7)
    print(result.shape)


if __name__ == '__main__':
    main()