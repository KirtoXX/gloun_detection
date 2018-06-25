"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from gluoncv.model_zoo.faster_rcnn.rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from gluoncv.model_zoo.rcnn import RCNN
from gluoncv.model_zoo.rpn import RPN
import mxnet.ndarray as nd
from mxnet import symbol

__all__ = ['LHRCNN', 'get_LHRCNN',
           'faster_rcnn_resnet50_v2a_voc',
           'faster_rcnn_resnet50_v2a_coco', ]

class Group_Conv(nn.HybridBlock):
    def __init__(self,output_dim=490):
        super(Group_Conv, self).__init__()
        self.conv1_1 = nn.Conv2D(256,(15,1),padding=(7,0),activation='relu')
        self.conv1_2 = nn.Conv2D(output_dim,(1,15),padding=(0,7))
        self.conv2_1 = nn.Conv2D(256,(1,15),padding=(0,7),activation='relu')
        self.conv2_2 = nn.Conv2D(output_dim,(15,1),padding=(7,0))
        self.bn = nn.BatchNorm()

    def forward(self, x, *args):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        out = self.bn(x1+x2)
        out = nd.relu(out)
        return out

class Head(nn.HybridBlock):
    def __init__(self,nb_class):
        super(Head, self).__init__()
        self.share = nn.Dense(2048,activation='relu')
        self.clf = nn.Dense(nb_class,activation='softmax')
        self.reg = nn.Dense(nb_class*4,activation='linear')

    def forward(self, x, *args):
        x = nd.Flatten(x)
        x = self.share(x)
        clf = self.clf(x)
        reg = self.reg(x)
        return clf,reg

class LHRCNN(RCNN):
    r"""Faster RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    train_patterns : str
        Matching pattern for trainable parameters.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh_high : float, default is 0.5
        Proposal whose IOU smaller than ``neg_iou_thresh_high``
        and larger than ``neg_iou_thresh_low`` is regarded as negative samples.
        Proposals with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    neg_iou_thresh_low : float, default is 0.0
        See ``neg_iou_thresh_high``.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.

    """
    def __init__(self, features, top_features, scales, ratios, classes, roi_mode, roi_size,
                 stride=16, rpn_channel=1024, num_sample=128, pos_iou_thresh=0.5,
                 neg_iou_thresh_high=0.5, neg_iou_thresh_low=0.0, pos_ratio=0.25, **kwargs):
        super(LHRCNN, self).__init__(
            features, top_features, classes, roi_mode, roi_size, **kwargs)
        self.stride = stride
        self._max_batch = 1  # currently only support batch size = 1
        self._max_roi = 100000  # maximum allowed ROIs
        self._target_generator = set([RCNNTargetGenerator(self.num_class)])
        with self.name_scope():
            self.rpn = RPN(rpn_channel, stride, scales=scales, ratios=ratios)
            self.sampler = RCNNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh_high,
                                            neg_iou_thresh_low, pos_ratio)
            self.head = Head(classes)
            k,_= self._roi_size
            self.group_conv = Group_Conv(10*k*k)


    @property
    def target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return list(self._target_generator)[0]

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Faster-RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """
        feat = self.features(x)
        feat2 = self.group_conv(feat)
        # RPN proposals
        if autograd.is_training():
            _, rpn_box, raw_rpn_score, raw_rpn_box, anchors = self.rpn(
                feat, F.zeros_like(x))
            # sample 128 roi
            assert gt_box is not None
            rpn_box, samples, matches = self.sampler(rpn_box, gt_box)
        else:
            _, rpn_box = self.rpn(feat, F.zeros_like(x))

        # create batchid for roi
        with autograd.pause():
            roi_batchid = F.arange(
                0, self._max_batch, repeat=self._max_roi).reshape(
                    (-1, self._max_roi))
            roi_batchid = F.slice_like(roi_batchid, rpn_box * 0, axes=(0, 1))
            rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)

        # ROI features
        '''
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rpn_roi, self._roi_size, 1. / self.stride)
        elif self._roi_mode == 'align':
            #print(self._roi_size)
            #print(1./self.stride)
            #print(rpn_roi.shape)
            pooled_feat = nd.ROIPooling(feat, rpn_roi, self._roi_size, 1. / self.stride)
            print(pooled_feat.shape)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
        '''
        pooled_feat = nd.contrib.PSROIPooling(data=feat2,
                                              rois=rpn_roi,
                                              spatial_scale=1. / self.stride,
                                              output_dim=10,
                                              pooled_size=self._roi_size)
        # RCNN prediction
        '''
        top_feat = self.top_features(pooled_feat)
        # top_feat = F.Pooling(top_feat, global_pool=True, pool_type='avg', kernel=self._roi_size)
        top_feat = self.global_avg_pool(top_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape(
            (-1, self.num_class, 4)).transpose((1, 0, 2))
        '''
        cls_pred,box_pred = self.head(pooled_feat)
        box_pred = box_pred.reshape((-1, self.num_class, 4)).transpose((1, 0, 2))
        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            box_pred = box_pred.transpose((1, 0, 2))
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)

        # translate bboxes
        bboxes = self.box_decoder(box_pred, self.box_to_center(rpn_box)).split(
            axis=0, num_outputs=self.num_class, squeeze_axis=True)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        results = []
        for i in range(self.num_class):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes[i]], dim=-1)

            results.append(per_result)
        result = F.concat(*results, dim=0).expand_dims(0)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                id_index=0, score_index=1, coord_start=2)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms).squeeze(axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        return ids, scores, bboxes

def get_LHRCNN(name, features, top_features, scales, ratios, classes,
               roi_mode, roi_size, dataset, stride=16,
               rpn_channel=1024, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    dataset : str
        The name of dataset.
    stride : int, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = LHRCNN(features, top_features, scales, ratios, classes, roi_mode, roi_size,
                 stride=stride, rpn_channel=rpn_channel, **kwargs)
    if pretrained:
        '''
        from ..model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
        '''
    return net


def My_LHRCNN(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    #>>> model = get_faster_rcnn_resnet50_v2_voc(pretrained=True)
    #>>> print(model)
    """
    from gluoncv.data import VOCDetection
    classes = VOCDetection.CLASSES
    my_class = []
    for i in range(60):
        my_class.append(str(i+1))
    '''
    if pretrained:
        pretrained_base = True
    else:
        pretrained_base = False
    '''
    from gluoncv.model_zoo.faster_rcnn.resnet50_v2a import resnet50_v2a
    #from ...data import COCODetection
    #classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v2a(pretrained=False)
    load_weight(base_network)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['rescale'] + ['layer' + str(i) for i in range(4)]:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*head','.*group_conv','.*rpn', '.*stage(2|3|4)_conv'])
    model = get_LHRCNN('resnet50_v2', features, top_features, scales=(2, 4, 8, 16, 32),
                      ratios=(0.5, 1, 2), classes=my_class, dataset='voc',
                      roi_mode='align', roi_size=(7,7), stride=16,
                      rpn_channel=1024, train_patterns=train_patterns,
                      pretrained=pretrained, **kwargs)
    print('build finish!')
    return model

def load_weight(model):
    from gluoncv.model_zoo.model_store import get_model_file
    model.load_params(get_model_file('resnet{}_v{}a'.format(50,2),
                                     root='pretrained/'), ctx=mx.cpu(), allow_missing=True)
    for v in model.collect_params(select='init_scale|init_mean').values():
        v.initialize(force_reinit=True)
    print('weight load finsih!')
    return model