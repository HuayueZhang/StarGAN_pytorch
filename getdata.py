import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
from torchvision import transforms

class myDataset(Dataset):
    def __init__(self, opt):
        super(myDataset, self).__init__()
        self.opt = opt
        self.train_dataset_list = []
        self.test_dataset_list = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.transfrom = []
        self.preprocess()
        self.get_transform()

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.opt.attr_path, 'r')]
        # string.rstrip() 返回删除string字符串末尾的制定字符（默认为空格）后生成的新字符串
        all_attr_names = lines[1].split()
        # string.split() 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等
        # 返回一个由切片后的字符串形成的列表
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2: ]
        random.seed(1234)
        random.shuffle(lines)
        num_train_batches = int(len(lines) * self.opt.spilt_ratio / self.opt.batch_size)
        num_train = num_train_batches * self.opt.batch_size
        num_test_batches = int((len(lines) - num_train) / self.opt.batch_size)
        num_test = num_test_batches * self.opt.batch_size

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1: ]

            label = []
            for attr in self.opt.selected_attr:
                label.append(values[self.attr2idx[attr]] == '1')
                # values是list，内容是选定特征的label，str形式，e.g. '-1', '1'
                # label转换value的格式'-1' -> False, '1' -> True

            if i < num_train:
                self.train_dataset_list.append([filename, label])
            elif i < (num_train + num_test):
                self.test_dataset_list.append([filename, label])
            else:
                break

    def get_transform(self):
        transform_list = []
        if self.opt.mode == 'train':
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.CenterCrop(self.opt.crop_size))
        transform_list.append(transforms.Resize(self.opt.image_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, idx):
        # if self.opt.mode == 'train':
        #     dataset = self.opt.train_dataset_list
        # else:
        #     dataset = self.opt.test_dataset_list
        dataset = self.train_dataset_list if self.opt.mode=='train' else self.test_dataset_list
        # 这是与上面if else 语句等价的简便写法
        filename, label = dataset[idx]
        img = Image.open(os.path.join(self.opt.dataroot, self.opt.dataset, filename))
        return self.transform(img), torch.FloatTensor(label)

    def __len__(self):
        if self.opt.mode == 'train':
            return len(self.train_dataset_list)
        else:
            return len(self.test_dataset_list)
