import os
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# my import
from dataset_all import TrainLabeled, TrainUnlabeled, ValLabeled
from UIE14_agg4 import *
# from UIE2 import *
# from Cross.dehazeformer import *
# from model import *
# from utils import *
# from model import *s
from trainer import Trainer


def main(gpu, args):  
    args.local_rank = gpu
    # random seed
    setup_seed(2022)
    # load data
    train_folder = args.data_dir

    # 加载带标签的训练数据集
    paired_dataset = TrainLabeled(dataroot=train_folder, phase='labeled', finesize=args.crop_size)

    # 加载无标签的训练数据集
    unpaired_dataset = TrainUnlabeled(dataroot=train_folder, phase='unlabeled', finesize=args.crop_size)

    # 加载验证数据集
    val_dataset = ValLabeled(dataroot=train_folder, phase='val', finesize=args.crop_size)

    paired_sampler = None
    unpaired_sampler = None
    val_sampler = None

    # 创建用于加载带标签的训练数据集的数据加载器
    paired_loader = DataLoader(paired_dataset, batch_size=args.train_batchsize, sampler=paired_sampler)

    # 创建用于加载无标签的训练数据集的数据加载器
    unpaired_loader = DataLoader(unpaired_dataset, batch_size=args.train_batchsize, sampler=unpaired_sampler)

    # 创建用于加载验证数据集的数据加载器
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, sampler=val_sampler)

    print('there are total %s batches for train' % (len(paired_loader)))
    print('there are total %s batches for val' % (len(val_loader)))

    # 创建模型
    # net = AIMnet()
    # net = UIE_org()
    # net = UIE_Third()
    # net = dehazeformer_s()
    net = UIE_Second()


    # ema_net = AIMnet()
    # ema_net = UIE_org()
    # ema_net = UIE_Third()
    # ema_net = dehazeformer_s()
    ema_net = UIE_Second()
    ema_net = create_emamodel(ema_net)

    print('student model params: %d' % count_parameters(net))

    # tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)

    # 创建训练器
    trainer = Trainer(model=net, tmodel=ema_net, args=args, supervised_loader=paired_loader,
                      unsupervised_loader=unpaired_loader,
                      val_loader=val_loader, iter_per_epoch=len(unpaired_loader), writer=writer)

    # 开始训练
    trainer.train()

    # 关闭tensorboard写入器
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')

    # 添加命令行参数
    parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--train_batchsize', default=3, type=int, help='train batchsize')
    parser.add_argument('--val_batchsize', default=1, type=int, help='val batchsize')
    parser.add_argument('--crop_size', default=256, type=int, help='crop size')
    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='./model/ckpt/best_in_NR.pth', type=str, help='if resume')
    parser.add_argument('--use_pretrain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained_path', default='./model/ckpt/best_in_NR.pth', type=str, help='if pretrained')
    parser.add_argument('--data_dir', default='./data/UIEBD2', type=str, help='data root path')
    parser.add_argument('--save_path', default='./model/ckpt/', type=str)
    parser.add_argument('--log_dir', default='./model/log', type=str)
    parser.add_argument('--lr', default='2e-4', type=float)
    parser.add_argument('--early_stop_epoch', default=200, type=int)

    args = parser.parse_args()

    # 如果保存路径不存在，则创建
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # 调用主函数进行训练
    main(-1, args)
