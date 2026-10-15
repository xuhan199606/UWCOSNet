import torch
import torch.nn.functional as F

import argparse
import os
import numpy as np
from tqdm import tqdm
import logging

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, eval_mae, numpy2tensor
from utils.loss import LossFunc
from utils.model import ModelFunc
from utils.output_list import OutputListFunc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 配置 logging
def setup_logging(log_file):
    # 创建一个日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器，用于在终端显示日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def test(model, data_path, infer, image_size):

    image_root = os.path.join(data_path, "Image")
    gt_root = os.path.join(data_path, "Mask")
    model.eval()
    num = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, image_size)
    mae_total = 0.0
    for i in tqdm(range(num)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        
        # inference and outlist process
        output = model(image)
        res = infer.calculate(output)
        
        # reshape and squeeze
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # evaluate
        mae = eval_mae(numpy2tensor(res), numpy2tensor(gt))

        mae_total += mae

    return mae_total / num


def train(train_loader, model, optimizer, lossfunc, epoch):
    model.train()

    total_step = len(train_loader)
    size_rates = [1] 
    loss_record = AvgMeter()
    i = 0 # 记录每个epoch到第几个batch了
    for pack in tqdm(train_loader):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = images.to(device)
            gts = gts.to(device)
            edges = edges.to(device)
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            output = model(images)
            # ---- loss function ----
            loss = lossfunc.calculate(output, gts, edges, i, total_step) 
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        i += 1 # 更新迭代i
        
    print('(Train) Epoch [{:03d}/{:03d}], Loss: {:0.4f}]'.
          format(epoch, opt.epoch, loss_record.show()))

def main(opt):
    # create save path
    save_work_dirs_path = opt.train_save + "_" + opt.datasets_name
    save_path = os.path.join(save_work_dirs_path, opt.model_name)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("Folder {} already exists, please check".format(save_path))
        exit()
    except Exception as e:
        print("Error creating folder:{}".format(save_path))
    
    # create log save
    log_file = os.path.join(save_path, "training.log")
    logger = setup_logging(log_file)

    # dataloader
    image_root = os.path.join(opt.train_path, "Image")
    gt_root = os.path.join(opt.train_path, "Mask")
    edge_root = os.path.join(opt.train_path, "Edge")

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    test_path = opt.test_path
    
    # create model
    model = ModelFunc(name=opt.model_name).to(device)

    # create optimizer
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        print("AdamW")
        optimizer = torch.optim.AdamW(params, opt.lr, betas=(0.9, 0.999), weight_decay=0.1)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=0.0005, momentum=0.9)


    # create loss
    lossfunc = LossFunc(name=opt.model_name)

    # train and eval
    print("#" * 20, "Start Training", "#" * 20)
    infer = OutputListFunc(name=opt.model_name)

    best = 10000000
    best_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
        train(train_loader, model, optimizer, lossfunc, epoch)
        mae = test(model, test_path, infer, image_size=opt.trainsize)

        if mae < best:
            best = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, opt.model_name+'-best.pth')) 


        logger.info('(Test) Epoch [{:03d}/{:03d}], MAE: {:0.4f}, best MAE: {:0.4f}, best epoch: {:0.4f}]'.
          format(epoch, opt.epoch, mae, best, best_epoch))    

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=100, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='Adam', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--datasets_name', type=str, default="ours")
    parser.add_argument('--train_path', type=str,default='', help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='', help='path to testing dataset')
    parser.add_argument('--train_save', type=str, default='./work_dirs')
    parser.add_argument('--model_name', type=str, default="ours")
    parser.add_argument('--device', type=int, default=0)

    opt = parser.parse_args()

    main(opt)

