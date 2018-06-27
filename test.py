from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
from gluoncv import model_zoo
from light_head_rcnn import My_LHRCNN
from gluoncv.model_zoo import model_zoo
from gluoncv import data
import matplotlib.pyplot as plt
from gluoncv import utils

def main():
    net = model_zoo.get_model('faster_rcnn_resnet50_v2a_voc')
    net.load_parameters('weights/frcnn.pkl')
    im_fname = 'data/train/img/00e93901213fb80e0f2eab4e3dd12f2eb83894a8.jpg'
    x,orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, box_ids, class_names=net.classes)
    plt.show()


if __name__ == '__main__':
    main()