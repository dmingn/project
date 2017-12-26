import random

import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable
from chainer.dataset import concat_examples


def generate_mask():
    width_w = random.randint(96, 128)
    width_h = random.randint(96, 128)
    offset_w = random.randint(0, 256 - width_w)
    offset_h = random.randint(0, 256 - width_h)
    range_w = range(offset_w, offset_w + width_w)
    range_h = range(offset_h, offset_h + width_h)
    return np.array([[i in range_w and j in range_h for i in range(256)] for j in range(256)]).astype(np.float32)


def mask_images(imgs, masks, mean_val):
    xp = chainer.cuda.get_array_module(imgs.data)
    return F.where(F.tile(F.expand_dims(F.cast(masks, bool), axis=1), (1, 3, 1, 1)), mean_val * xp.ones(imgs.shape, dtype=xp.float32), imgs)


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.mean_val = kwargs.pop('mean_val')
        self.t_c = kwargs.pop('t_c')
        self.t_d = kwargs.pop('t_d')
        self.alpha = kwargs.pop('alpha')
        super(Updater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, imgs_masked, imgs_completed, masks):
        loss = F.mean(F.batch_l2_norm_squared(
            F.tile(F.expand_dims(masks, axis=1), (1, 3, 1, 1)) * (imgs_masked - imgs_completed)))
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_dis(self, dis, d_fake, d_real):
        loss = -F.mean((F.log(d_real) + F.log(1 - d_fake)))
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_joint(self, gen, alpha, imgs_masked, imgs_completed, masks, d_fake, d_real):
        loss1 = F.mean(F.batch_l2_norm_squared(
            F.tile(F.expand_dims(masks, axis=1), (1, 3, 1, 1)) * (imgs_masked - imgs_completed)))
        loss2 = alpha * F.mean((F.log(d_real) + F.log(1 - d_fake)))
        loss = loss1 + loss2
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        # Sample minibatch
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        imgs_orig = Variable(self.converter(batch, self.device))

        # Generate a random hole in the [96, 128] pixel range
        masks_gen = Variable(concat_examples(
            [generate_mask() for i in range(batchsize)], self.device))

        # Fill the holes
        imgs_masked = mask_images(imgs_orig, masks_gen, self.mean_val)

        gen, dis = self.gen, self.dis

        if self.iteration < self.t_c:
            imgs_completed = gen(imgs_masked)
            gen_optimizer.update(self.loss_gen, gen, imgs_masked, imgs_completed, masks_gen)
        else:
            imgs_completed = gen(imgs_masked)
            d_fake = dis(imgs_completed, masks_gen)
            masks_dis = Variable(concat_examples(
                [generate_mask() for i in range(batchsize)], self.device))
            d_real = dis(imgs_orig, masks_dis)
            dis_optimizer.update(self.loss_dis, dis, d_fake, d_real)

            if self.iteration > self.t_c + self.t_d:
                gen_optimizer.update(self.loss_joint, gen, self.alpha, imgs_masked, imgs_completed, masks_gen, d_fake, d_real)
