from tensorboardX import SummaryWriter
import os
import utils
import numpy as np
from PIL import Image

class TBVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.save_path = os.path.join(opt.save_dir, opt.log_folder)

        self.log_path = os.path.join(self.save_path, 'loss_log.txt')
        self.tb_path = os.path.join(self.save_path, 'summaty.josn')
        self.writer = SummaryWriter(self.save_path)

        with open(self.log_path, "a") as log_file:
            log_file.write('============ Training Loss =============')

    def __del__(self):
        self.writer.close()

    def scalar(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step)
        self.writer.export_scalars_to_json(self.tb_path)

    def image(self, tag, image, global_step, opt):
        self.writer.add_image(tag, image, global_step, dataformats='HWC')

        # save image
        image = image * 255.
        image = image.astype(np.uint8)
        manifold_img_PIL = Image.fromarray(image)

        filename = 'sample_iter_%d.png' % global_step
        savepath = os.path.join(opt.save_dir, opt.sample_folder, filename)
        manifold_img_PIL.save(savepath)


    def log_and_print(self, message):
        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write('%s\n' % message)