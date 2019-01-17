import numpy as np
import torch

def get_manifold_img_array(real_images, fake_images_list, opt):  # 5 * (16, 3, 128, 128)
    num_domain = len(fake_images_list)
    num_image = fake_images_list[0].size(0)
    w0 = opt.image_size
    h0 = opt.image_size
    grid = np.zeros((3, h0 * num_image, w0 * (num_domain+1)))  # (3, H, W)

    # the first column is real images
    for j, image in enumerate(real_images):
        # image = image.numpy()
        grid[:, h0*j: h0*(j+1), 0: w0] = image.detach()

    # each column is a domain consisting of 16 images
    for i, fake_images in enumerate(fake_images_list):  # fake_images (16, 3, 128, 128)
        for j, image in enumerate(fake_images):         # image  (3, 128, 128)
            # image = image.numpy()
            grid[:, h0*j: h0*(j+1), w0*(i+1): w0*(i+2)] = image.detach()

    grid = np.transpose(grid, (1, 2, 0))  # (H, W, 3)
    return grid