from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd
from gluoncv.data.batchify import Tuple, Append
from mxnet.gluon.data import DataLoader
from dataset import Dataset

def main():
    short, max_size = 800, 1000  # resize image to short side 600 px, but keep maximum length within 1000
    train_transform = presets.rcnn.FasterRCNNDefaultTrainTransform(short, max_size)
    batch_size = 2  # for tutorial, we use smaller batch-size
    num_workers = 2  # you can make it larger(if your CPU has more cores) to accelerate data loading

    train_dataset = Dataset()
    batchify_fn = Tuple(Append(), Append())
    train_dataset.transform = train_transform
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)

    for ib, batch in enumerate(train_loader):
        if ib > 3:
            break
        print('data 0:', batch[0][0].shape, 'label 0:', batch[1][0].shape)
        print('data 1:', batch[0][1].shape, 'label 1:', batch[1][1].shape)


if __name__ == '__main__':
    main()

