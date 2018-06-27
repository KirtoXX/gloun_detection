from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
from gluoncv import model_zoo
from light_head_rcnn import My_LHRCNN
from faster_rcnn import Faster_rcnn
from gluoncv import data
import matplotlib.pyplot as plt
from gluoncv import utils

def main():
    net = Faster_rcnn(pretrained_base=False)
    net.load_parameters('weights/frcnn_0.pkl')
    im_fname = 'data/train/img/00e93901213fb80e0f2eab4e3dd12f2eb83894a8.jpg'
    x,orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.plot_bbox(orig_img, bboxes, scores, box_ids)
    plt.show()


if __name__ == '__main__':
    main()