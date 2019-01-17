from torch import nn
import torch
import numpy as np

class ResidualBlock(nn.Module):
    # 不只是G和D网络，写任何一个网络，都要按这种继承的方式写
    def __init__(self, c_in, c_out):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(c_out, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(c_out, affine=True, track_running_stats=True))

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input) + input


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        conv_dim = opt.G_filter_size

        layers = []
        layers.append(nn.Conv2d(3+opt.num_attr, conv_dim, kernel_size=7, stride=1, padding=3))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # down-sampling layers
        for i in range(2):
            layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim = conv_dim * 2

        # bottleneck layers
        for i in range(opt.num_repeat):
            layers.append(ResidualBlock(c_in=conv_dim, c_out=conv_dim))

        # up-sampling layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(conv_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            conv_dim = conv_dim // 2

        layers.append(nn.ConvTranspose2d(conv_dim, 3, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())
        # 在sequential中包装的是nn.module的子类，其他操作不能包装到sequential中，比如view(reshape)操作

        self.main = nn.Sequential(*layers)

    def forward(self, images, c_trg):
        # construct input of G by concat images and labels
        # images(b, c, h, w), labels(b, num_attr)
        # Replicate spatially and concatenate domain information.
        # label是通过叠加到每个像素点后面，实现conditional generate，有目的性的生成图像
        c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)  # (b, num_attr, 1, 1)
        c_trg = c_trg.repeat(1, 1, images.size(2), images.size(3))# (b, num_attr, h, w)
        inputs = torch.cat([images, c_trg], dim=1)               # (b, c+num_attr, h, w)
        G_imgs = self.main(inputs)
        return G_imgs


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        conv_dim = opt.D_filter_size

        layers = []
        # input layer
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        # repeated hidden layers
        for i in range(1, opt.num_repeat):
            layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            conv_dim = conv_dim * 2

        # output layers
        self.main = nn.Sequential(*layers)
        self.src = nn.Conv2d(conv_dim, 1, kernel_size=3, stride=1, padding=1)
        ksize = int(opt.image_size // np.power(2, opt.num_repeat))
        self.cls = nn.Conv2d(conv_dim, opt.num_attr, kernel_size=ksize, stride=1, padding=0)
        # 要用到image size这种数，要通过参数传递进来，反正__init__里面是没有真实image数据传进来的，
        # 无法通过image.size这种方式获得
        # 并且网络层又一定要在__init__里面，不能放到forward里面，虽然forward里面有真实的image

    def forward(self, images):
        h = self.main(images)
        out_src = self.src(h)
        out_cls = self.cls(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
