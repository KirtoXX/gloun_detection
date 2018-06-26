from gluoncv.data import VOCDetection
from gluoncv.data.base import VisionDataset
import mxnet as mx
import os
import numpy as np
from matplotlib import pyplot as plt
from gluoncv.utils import viz

def ann_to_list(ann_path):
    label_list = []
    f = open(ann_path ,mode='r')
    for line in f:
        c ,x1 ,y1 ,x2 ,y2 = line.split(' ')
        label_list.append([int(x1) ,int(y1) ,int(x2) ,int(y2),int(c)])
    return label_list

class Dataset(VisionDataset):
    def _list_name(self):
        file_list = os.listdir('data/train/img')
        return file_list

    def __init__(self,transform=None):
        self.file = self._list_name()
        self._transform = transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, item):
        name = self.file[item]
        img_path = 'data/train/img/{}'.format(name)
        ann_path = 'data/train/ann/{}.txt'.format(name)
        img = mx.img.imread(img_path,1)
        label = ann_to_list(ann_path)
        if self._transform is not None:
            return self._transform(img,label)
        label = np.array(label)
        return img,label


def main():
    train_dataset = Dataset()
    train_image, train_label = train_dataset[0]
    bboxes = train_label[:, :4]
    cids = train_label[:, 4:5]
    print('image:', train_image.shape)
    print('bboxes:', bboxes.shape, 'class ids:', cids.shape)
    ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids)
    plt.show()



if __name__ == '__main__':
    main()
