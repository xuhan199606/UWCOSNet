import torch
import torch.nn as nn
import torch.nn.functional as F


"""
pre, gts, edges, i=None, total_step=None

pre: outpur for model
gts: ground truth for mask
edges: ground truth for edge
i: the step for each epoch
total_step: total step for each
"""


LogitsBCE = torch.nn.BCEWithLogitsLoss()

bce = nn.BCELoss(reduction='mean')
mse = nn.MSELoss(reduction='mean')


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def iou_loss(pred, target):

    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

def bce_iou_loss(pred, target):
    bce_out = LogitsBCE(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


class LossFunc:
    def __init__(self, name):
        if name == "model_name":
            print("self.calculate = self.Func")
        elif name == "UWCOSNet_ResNet50":
            self.calculate = self.UWCOSNet_Func
        elif name == "UWCOSNet_PVTv2B2":
            self.calculate = self.UWCOSNet_Func
        

    
    def UWCOSNet_Func(self, pre, gts, edges, i=None, total_step=None):
        out1, out2, out3, out4, out_global_mask, out_global_edge, out_edge1, out_edge2, out_edge3, out_edge4 =  pre
        loss1 = structure_loss(out1, gts)
        loss2 = structure_loss(out2, gts)
        loss3 = structure_loss(out3, gts)
        loss4 = bce_iou_loss(out4, gts)

        loss_gm = structure_loss(out_global_mask, gts)
        loss_ge = dice_loss(out_global_edge, edges)

        loss_edge1 = dice_loss(out_edge1, edges)
        loss_edge2 = dice_loss(out_edge2, edges)
        loss_edge3 = dice_loss(out_edge3, edges)
        loss_edge4 = dice_loss(out_edge4, edges)
        
        loss = loss1 + loss2 + loss3 + loss4 + loss_gm + loss_ge + loss_edge1 + loss_edge2 + loss_edge3 + loss_edge4

        return loss


if __name__ == "__main__":
    loss = LossFunc("UWCOSNet_ResNet50")
    loss.calculate()