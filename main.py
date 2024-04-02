#!/home/liuyang/anaconda3/envs/pytorch/bin/python
# _*_ coding: utf-8 _*_
import torch
import argparse
import sys

sys.path.append('./')
from Application import App


def main(args):
    if args.classifer == 'train':
        # print('into train prosses')
        # train(args)
        mask = App(args)
        mask.train()
    # elif args.classifer == 'test':
    #     print('into test prosses')
    #     test(args)
    elif args.classifer == 'predict':
        print('into predict prosses')

    else:
        print('No good define for parser, check your command please!')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='face quality classfier, for bad face_img and good face_img')
    # train prosess
    parser.add_argument('--train_id', type=str, default="2")
    parser.add_argument('--classifer', type=str, default='train',
                        help='three choose: train: for train net \ntest: for test net\n data: data prosessing ')
    parser.add_argument('--mode_path', type=str, default='./models/')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_class', type=int, default=37)
    parser.add_argument('--gpus', type=list, default=[0])
    parser.add_argument('--lr_step', type=int, default=50)
    # model change
    parser.add_argument('--model_name', type=str, default='pets')
    parser.add_argument('--valid_percent', type=float, default=0.2)
    parser.add_argument('--valid_per_epoch', type=int, default=2)
    parser.add_argument('--cls_loss_weight', type=float, default=0.3)

    # data prosess
    parser.add_argument('--data_path', type=str, default='/home/wsx/ptJobs/TB20201576A5/data/')
    parser.add_argument('--version', type=str, default='version 1.0')

    # model prosess
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_epoch', type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)
