import torch
from net import Discriminator, Generator
from utils import get_manifold_img_array
from visualizer import TBVisualizer
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np

class StarGAN:
    def __init__(self, opt):
        self.opt = opt
        self.global_step = opt.load_iter
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define net class instance
        self.G_net = Generator(opt).to(self.device)
        self.D_net = Discriminator(opt).to(self.device)
        if opt.load_model and opt.load_iter > 0:
            self._load_pre_model(self.G_net, 'G')
            self._load_pre_model(self.D_net, 'D')

        # define objectives and optimizers
        self.adv_loss = torch.mean   # 这里的adv loss直接是真假结果的评分，真图越大越好，假图越小越好
        # self.cls_loss = torch.nn.BCELoss() # ??????????? 有啥区别
        self.cls_loss = F.binary_cross_entropy_with_logits
        self.rec_loss = torch.mean
        self.G_optimizer = torch.optim.Adam(self.G_net.parameters(), opt.G_lr, [opt.beta1, opt.beta2])
        self.D_optimizer = torch.optim.Adam(self.D_net.parameters(), opt.D_lr, [opt.beta1, opt.beta2])

        self.sample_gotten = False  # 把它放在init里面，是因为它只随着类的调用初始化一次，是固定的sample
        self.writer = TBVisualizer(opt)

    def _load_pre_model(self, net, module):
        filename = '%d-%s.ckpt' % (self.opt.load_iter, module)
        loadpath = os.path.join(self.opt.save_dir, self.opt.model_folder, filename)
        net.load_state_dict(torch.load(loadpath))
        print('load model: %s' % loadpath)

    def _set_eval_sample(self):
        # let the sample be the first batch of the whole training process
        self.sample_real = self.img_real
        self.sample_c_trg_list= self._create_fix_trg_label(self.c_org)
        self.sample_gotten = True

    def _create_fix_trg_label(self, c_org):
        # eval的时候，希望看到固定初始图片被转换成固定的其他样子=>设置固定的target domain labels
        # test的时候，也要这样为测试图片选择固定的目标域
        hair_color_ids = []
        for id, selected_attr in enumerate(self.opt.selected_attr):
            if selected_attr in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_ids.append(id)

        c_trg = c_org.clone()
        c_trg_list = []
        for i in range(self.opt.num_attr):
            # 把由5个特征（头发0，头发1，头发2，性别，年龄）表示的original domain转换到5个由5个特征表示的target domain，
            # target domain1：拥有头发0，性别年龄不变
            # target domain2：拥有头发1，性别年龄不变
            # target domain3：拥有头发2，性别年龄不变
            # target domain4：头发不变，性别改变，年龄不变
            # target domain5：头发不变，性别不变，年龄改变
            if i in hair_color_ids:
                for j in hair_color_ids:
                    c_trg[:, j] = int(i == j)
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)

            c_trg_list.append(c_trg)
        return c_trg_list

    def set_inputs(self, data):
        img_real, labels_org = data
        self.img_real = img_real.to(self.device)
        self.c_org = labels_org.to(self.device)
        # generate target domain labels randomly
        # torch.randperm(n)把1到n这些数随机打乱得到的一个数字序列
        rand_idx = torch.randperm(labels_org.size(0))
        labels_trg = labels_org[rand_idx]
        self.c_trg = labels_trg.to(self.device)

    def _optimize_D(self, visualize=True):
        # forward
        # using real images
        src_real, cls_real = self.D_net(self.img_real)
        loss_D_real = - self.adv_loss(src_real)  # loss都是越小越好，如果希望越大越好，就取负
        loss_D_cls = self.cls_loss(cls_real, self.c_org)

        # using fake image
        img_fake = self.G_net(self.img_real, self.c_trg)
        src_fake, cls_fake = self.D_net(img_fake.detach())
        loss_D_fake = self.adv_loss(src_fake)

        loss_D = loss_D_real + loss_D_fake + self.opt.cls_lambda * loss_D_cls

        # backward
        self.D_optimizer.zero_grad()
        loss_D.backward()
        self.D_optimizer.step()

        # if visualize:
        self.writer.scalar('loss_D_real', loss_D_real, self.global_step)
        self.writer.scalar('loss_D_fake', loss_D_fake, self.global_step)
        self.writer.scalar('loss_D_cls', loss_D_fake, self.global_step)
        self.writer.scalar('loss_D', loss_D, self.global_step)

        return loss_D

    def _optimize_G(self, visualize=True):
        # forward
        img_fake = self.G_net(self.img_real, self.c_trg)

        # fuse discriminator
        src_fake, cls_fake = self.D_net(img_fake)
        loss_G_fake = - self.adv_loss(src_fake)
        loss_G_cls = self.cls_loss(cls_fake, self.c_trg)

        # reconstruct images
        img_rec = self.G_net(img_fake, self.c_org)
        loss_G_rec = self.rec_loss(torch.abs(img_rec-self.img_real))

        loss_G = loss_G_fake + self.opt.cls_lambda * loss_G_cls + self.opt.rec_lambda * loss_G_rec

        # backward
        self.G_optimizer.zero_grad()
        loss_G.backward()
        self.G_optimizer.step()

        # if visualize:
        self.writer.scalar('loss_G_fake', loss_G_fake, self.global_step)
        self.writer.scalar('loss_G_cls', loss_G_cls, self.global_step)
        self.writer.scalar('loss_G_rec', loss_G_rec, self.global_step)
        self.writer.scalar('loss_G', loss_G, self.global_step)
        return loss_G

    def eval_sample(self):
        if not self.sample_gotten:
            self._set_eval_sample()
        sample_fake_list = []
        for sample_c_trg in self.sample_c_trg_list:
            sample_fake = self.G_net(self.sample_real, sample_c_trg)  # (16, 5, 128, 128)
            sample_fake_list.append(sample_fake)

        return sample_fake_list

    def optimizer(self):
        self.global_step += 1
        loss_D = self._optimize_D(visualize=True)
        # loss_G = self._optimize_G(visualize=False)
        loss_G = self._optimize_G(visualize=True)

        if self.global_step % 100 == 0:
            message = 'iter[%6d/%d], loss_D = %.6f, loss_G = %.6f' % \
                      (self.global_step, self.opt.max_iter, loss_D, loss_G)
            self.writer.log_and_print(message)

        if self.global_step % 1000 == 0:
            sample_fake_list = self.eval_sample()
            manifold_img_array = get_manifold_img_array(self.sample_real, sample_fake_list, self.opt)  # (H, W, 3)
            manifold_img_array = (manifold_img_array + 1.) / 2.
            self.writer.image('eval_sample', manifold_img_array, self.global_step, self.opt)

        if self.global_step % 10000 == 0:
            self.save_model()

    # def test(self):

    def save_model(self):
        filename = '%d-G.ckpt' % self.global_step
        savepath = os.path.join(self.opt.save_dir, self.opt.model_folder, filename)
        torch.save(self.G_net.state_dict(), savepath)

        filename = '%d-D.ckpt' % self.global_step
        savepath = os.path.join(self.opt.save_dir, self.opt.model_folder, filename)
        torch.save(self.D_net.state_dict(), savepath)