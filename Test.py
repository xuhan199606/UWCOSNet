import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import cv2
from tqdm import tqdm

from utils.dataloader import test_dataset
from utils.model import ModelFunc
from utils.model_load import ModelLoadFunc
from utils.output_list import OutputListFunc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='SINet-best.pth')
    parser.add_argument('--model_name', type=str, default='write')
    parser.add_argument('--data_path', type=str, default='MAS3K-Camouflaged')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_name', type=str)
    opt = parser.parse_args()

    # create save path
    save_path = os.path.join(opt.save_path, opt.data_name, opt.model_name)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("文件夹{}已存在，请检查".format(save_path))
        exit()
    except Exception as e:
        print("创建文件夹时出错：{}".format(save_path))
    model = ModelFunc(name=opt.model_name)
    model = ModelLoadFunc(name=opt.model_name, model=model, pth_path=opt.pth_path)
    model.to(device)
    model.eval()
    print("create model")
    
    infer = OutputListFunc(name=opt.model_name)


    image_root = os.path.join(opt.data_path, "Image")
    gt_root = os.path.join(opt.data_path, "Mask")

    # num = len(os.listdir(gt_root))
    print(image_root)
    num = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    # create data_name 
    save_data_path = os.path.join(save_path)
    
    for i in tqdm(range(num)):
        image, gt, image_name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        output = model(image)
        res = infer.calculate(output)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print(os.path.join(save_data_path, image_name))
        # print(image_name)
        cv2.imwrite(os.path.join(save_data_path, image_name), res*255)
  
