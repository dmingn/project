import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable
from chainer.dataset import concat_examples


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()

        with self.init_scope():
            self.l1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=1, pad=2)
            self.l2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=2, pad=1)
            self.l3 = L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1)
            self.l4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=2, pad=1)
            self.l5 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1)
            self.l6 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1)
            self.l7 = L.DilatedConvolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=2, dilate=2)
            self.l8 = L.DilatedConvolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=4, dilate=4)
            self.l9 = L.DilatedConvolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=8, dilate=8)
            self.l10 = L.DilatedConvolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=16, dilate=16)
            self.l11 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1)
            self.l12 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1)
            self.l13 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=4, stride=2, pad=1)
            self.l14 = L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1)
            self.l15 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=4, stride=2, pad=1)
            self.l16 = L.Convolution2D(in_channels=64, out_channels=32, ksize=3, stride=1, pad=1)
            self.l17 = L.Convolution2D(in_channels=32, out_channels=3, ksize=3, stride=1, pad=1)

            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(256)
            self.bn5 = L.BatchNormalization(256)
            self.bn6 = L.BatchNormalization(256)
            self.bn7 = L.BatchNormalization(256)
            self.bn8 = L.BatchNormalization(256)
            self.bn9 = L.BatchNormalization(256)
            self.bn10 = L.BatchNormalization(256)
            self.bn11 = L.BatchNormalization(256)
            self.bn12 = L.BatchNormalization(256)
            self.bn13 = L.BatchNormalization(128)
            self.bn14 = L.BatchNormalization(128)
            self.bn15 = L.BatchNormalization(64)
            self.bn16 = L.BatchNormalization(32)

    def __call__(self, imgs):
        h = F.relu(self.bn1(self.l1(imgs)))
        h = F.relu(self.bn2(self.l2(h)))
        h = F.relu(self.bn3(self.l3(h)))
        h = F.relu(self.bn4(self.l4(h)))
        h = F.relu(self.bn5(self.l5(h)))
        h = F.relu(self.bn6(self.l6(h)))
        h = F.relu(self.bn7(self.l7(h)))
        h = F.relu(self.bn8(self.l8(h)))
        h = F.relu(self.bn9(self.l9(h)))
        h = F.relu(self.bn10(self.l10(h)))
        h = F.relu(self.bn11(self.l11(h)))
        h = F.relu(self.bn12(self.l12(h)))
        h = F.relu(self.bn13(self.l13(h)))
        h = F.relu(self.bn14(self.l14(h)))
        h = F.relu(self.bn15(self.l15(h)))
        h = F.relu(self.bn16(self.l16(h)))
        h = F.sigmoid(self.l17(h))
        return h


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()

        with self.init_scope():
            self.l1_l = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=2, pad=2)
            self.l2_l = L.Convolution2D(in_channels=64, out_channels=128, ksize=5, stride=2, pad=2)
            self.l3_l = L.Convolution2D(in_channels=128, out_channels=256, ksize=5, stride=2, pad=2)
            self.l4_l = L.Convolution2D(in_channels=256, out_channels=512, ksize=5, stride=2, pad=2)
            self.l5_l = L.Convolution2D(in_channels=512, out_channels=512, ksize=5, stride=2, pad=2)
            self.l6_l = L.Linear(in_size=None, out_size=1024)

            self.bn1_l = L.BatchNormalization(64)
            self.bn2_l = L.BatchNormalization(128)
            self.bn3_l = L.BatchNormalization(256)
            self.bn4_l = L.BatchNormalization(512)
            self.bn5_l = L.BatchNormalization(512)

            self.l1_g = L.Convolution2D(in_channels=3, out_channels=64, ksize=5, stride=2, pad=2)
            self.l2_g = L.Convolution2D(in_channels=64, out_channels=128, ksize=5, stride=2, pad=2)
            self.l3_g = L.Convolution2D(in_channels=128, out_channels=256, ksize=5, stride=2, pad=2)
            self.l4_g = L.Convolution2D(in_channels=256, out_channels=512, ksize=5, stride=2, pad=2)
            self.l5_g = L.Convolution2D(in_channels=512, out_channels=512, ksize=5, stride=2, pad=2)
            self.l6_g = L.Convolution2D(in_channels=512, out_channels=512, ksize=5, stride=2, pad=2)
            self.l7_g = L.Linear(in_size=None, out_size=1024)

            self.bn1_g = L.BatchNormalization(64)
            self.bn2_g = L.BatchNormalization(128)
            self.bn3_g = L.BatchNormalization(256)
            self.bn4_g = L.BatchNormalization(512)
            self.bn5_g = L.BatchNormalization(512)
            self.bn6_g = L.BatchNormalization(512)

            self.l_c = L.Linear(in_size=None, out_size=1)

    def __call__(self, imgs, masks):
        xp = chainer.cuda.get_array_module(masks.data)
        centers = [[int(a.mean()) for a in xp.where(mask.data == 1)] for mask in masks]
        slices = [
            (..., *tuple(
                slice(max(0, c - 64) + min(0, m - c - 64),
                      min(m, c + 64) + max(0, 64 - c))
                for (c, m) in zip(center, imgs.shape[2:4])
            )) for center in centers]
        imgs_l = F.concat(tuple(F.expand_dims(F.get_item(img, s), axis=0) for (img, s) in zip(imgs, slices)), axis=0)
        h_l = F.relu(self.bn1_l(self.l1_l(imgs_l)))
        h_l = F.relu(self.bn2_l(self.l2_l(h_l)))
        h_l = F.relu(self.bn3_l(self.l3_l(h_l)))
        h_l = F.relu(self.bn4_l(self.l4_l(h_l)))
        h_l = F.relu(self.bn5_l(self.l5_l(h_l)))
        h_l = self.l6_l(h_l)

        imgs_g = F.resize_images(imgs, (256, 256))
        h_g = F.relu(self.bn1_g(self.l1_g(imgs_g)))
        h_g = F.relu(self.bn2_g(self.l2_g(h_g)))
        h_g = F.relu(self.bn3_g(self.l3_g(h_g)))
        h_g = F.relu(self.bn4_g(self.l4_g(h_g)))
        h_g = F.relu(self.bn5_g(self.l5_g(h_g)))
        h_g = F.relu(self.bn6_g(self.l6_g(h_g)))
        h_g = self.l7_g(h_g)

        h = F.sigmoid(self.l_c(F.concat((h_l, h_g), axis=1)))
        return h
