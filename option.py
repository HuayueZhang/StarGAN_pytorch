import argparse
import os
import torch

class BaseOption:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train or test')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu idx: e.g., 0 | 0,1 | 0,1,2 | use -1 for cpu')
        parser.add_argument('--num_workers', type=int, default=2, help='number of threads')

        # training related option
        parser.add_argument('--batch_size', type=int, default=16, help='mini batch size')
        parser.add_argument('--crop_size', type=int, default=178, help='crop image into 178 x 178')
        parser.add_argument('--image_size', type=int, default=128, help='resize image into 128 x 128')
        parser.add_argument('--max_iter', type=int, default=200000, help='max iterations in training')
        parser.add_argument('--G_lr', type=float, default=0.0001, help='learning rate for G')
        parser.add_argument('--D_lr', type=float, default=0.0001, help='learning rate for D')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
        parser.add_argument('--load_model', type=bool, default=True, help='load pretrained model is True')
        parser.add_argument('--load_iter', type=int, default=0, help='load the latest pretrained model')
        parser.add_argument('--cls_lambda', type=int, default=1, help='coefficient of cls loss')
        parser.add_argument('--rec_lambda', type=int, default=10, help='coefficient of rec loss')

        # net parameters
        parser.add_argument('--G_filter_size', type=int, default=64, help='the size of the first filter of G net')
        parser.add_argument('--D_filter_size', type=int, default=64, help='the size of the first filter of D net')
        parser.add_argument('--num_repeat', type=int, default=6, help='number of repeated blocks in net')

        # dataset related option
        parser.add_argument('--dataroot', type=str, default='/home/zhy/myworks/DCGAN/data', help='root dir of all datasets')
        parser.add_argument('--dataset', type=str, default='celebA', help='choose dataset: celebA | zhy | mnist')
        parser.add_argument('--attr_path', type=str, help='the attributes of celaba during training',
                            default='/home/zhy/myworks/stargan/data/celebA/Anno/list_attr_celeba.txt')
        parser.add_argument('--spilt_ratio', type=float, default='0.8', help='ratio used to split train and test dataset')
        parser.add_argument('--selected_attr', '--list', nargs='+', help='selected attributes for training',
                            default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
        parser.add_argument('--num_attr', type=int, default=5, help='number of selected attributes')


        # model and log saving path
        parser.add_argument('--save_dir', type=str, help='directory to save model, log, result, etc',
                            default='/home/zhy/myworks/stargan_pytorch/save/')
        parser.add_argument('--model_folder', type=str, default='model', help='train to save model')
        parser.add_argument('--log_folder', type=str, default='logs', help='train: save logs')
        parser.add_argument('--result_folder', type=str, default='results', help='test: save results')
        parser.add_argument('--sample_folder', type=str, default='samples', help='train: save samples')

        self.initialized = True
        return parser

    def parse(self):
        if self.initialized == False:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()

        opt.num_attr = len(opt.selected_attr)

        # check saving paths and create if not exist
        if not os.path.exists(opt.save_dir):
            os.mkdir(opt.save_dir)

        model_path = os.path.join(opt.save_dir, opt.model_folder)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            opt.load_model = False

        log_path = os.path.join(opt.save_dir, opt.log_folder)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        result_path = os.path.join(opt.save_dir, opt.result_folder)
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        sample_path = os.path.join(opt.save_dir, opt.sample_folder)
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)

        # get iter of loaded model
        if opt.load_model:
            load_iter = 0
            for filename in os.listdir(model_path):
                if filename.endswith('.ckpt'):
                    split = filename.split('-')
                    load_iter = max(load_iter, int(split[0]))
            opt.load_iter = load_iter

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        print(opt)
        self.opt = opt
        return self.opt