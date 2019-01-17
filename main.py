from option import BaseOption
from getdata import myDataset
from model import StarGAN
from torch.utils.data import DataLoader

def main():
    import setproctitle
    setproctitle.setproctitle('py3')

    opt = BaseOption().parse()

    myData = myDataset(opt)
    myDataLoader = DataLoader(myData,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers)
    # train
    model = StarGAN(opt)
    print('Start training . . .')
    while(model.global_step < opt.max_iter):
    #for epoch in range(pre_epoch, opt.max_epoch):
        for batch, data in enumerate(myDataLoader):
            model.set_inputs(data)
            model.optimizer()



if __name__ == '__main__':
    main()