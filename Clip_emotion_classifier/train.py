# -*- coding: utf-8 -*-
import math
import argparse
import datetime
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dataset import *
import random
import numpy as np
from model import BackBone as model_cnn
from utils import train_one_epoch, evaluate
from transformers import CLIPModel, CLIPProcessor


# random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.benchmark = False                   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # cudnn
    # os.environ['PYTHONHASHSEED'] = str(seed)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    best_epoch = 0
    best_acc = 0
    set_seed(seed=opt.seed)
    CLIPmodel = CLIPModel.from_pretrained("/mnt/d/model/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
    processor = CLIPProcessor.from_pretrained("/mnt/d/model/CLIP-ViT-H-14-laion2B-s32B-b79K")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 实例化训练数据集
    train_dataset = EmoSet(args.data_path, 'train',model=CLIPmodel, processor=processor)
    # 实例化验证数据集
    val_dataset = EmoSet(args.data_path, 'val',model=CLIPmodel, processor=processor)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               #   collate_fn=train_dataset.collate_fn
                                               drop_last=False
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=nw,
                                             #  collate_fn=val_dataset.collate_fn
                                             drop_last=False
                                             )

    model = model_cnn().to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # head 写入
    with open('./weights/train_detail.txt', 'a') as f:
        f.write(str(datetime.date.today()) + ', lr, epoch, train_loss, train_acc, val_loss, val_acc\n')
    # 记录loss
    with open('./weights/test_loss_detail.txt', 'a') as f:
        f.write(str(datetime.date.today()) + ',epoch,{acc_loss_dic},(step + 1)\n')

    for epoch in range(args.epochs):
        print('lr: ', args.lr)
        print('batch_size: ', args.batch_size)
        print('total_epoch: ', args.epochs)
        pth_path = './weights/' + str(datetime.date.today()) + '-best.pth'

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lable_mode=args.lable_mode,
                                                # NO_lable=args.NOL,
                                                CLIPmodel=CLIPmodel)

        scheduler.step()  # 调整lr

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     lable_mode=args.lable_mode,
                                     # NO_lable=args.NOL,
                                     CLIPmodel=CLIPmodel)
        # Best acc  
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), pth_path)

        now = datetime.datetime.now()
        time_str = now.strftime('%Y-%m-%d_%H-%M')
        output_dir = f'time_{time_str}'
        pth_path = './weights/' + output_dir + '-best.pth'
        torch.save(model.state_dict(), pth_path)
        print('best_epoch: % .0f' % best_epoch, 'best_acc: % .3f' % best_acc)
        with open('./weights/train_detail.txt', 'a') as f:
            f.write(str(datetime.date.today())+','+str(args.lr)+','+str(epoch)+','+str(train_loss)+','
                    +str(train_acc)+','+str(val_loss)+','+str(val_acc)+','+str(best_acc)+'\n')

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--NOL', type=str, default=-1, help='No lable mark')

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data_path', type=str,
                        default='/mnt/d/dataset/EmoSet_v5_train-test-val')
    parser.add_argument('--model_name', default='', help='create model name')
    parser.add_argument('--lable_mode', type=str, default='emotion')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', default=66, type=int, help='just a random seed')
    torch.multiprocessing.set_start_method('spawn')
    opt = parser.parse_args()

    main(opt)
