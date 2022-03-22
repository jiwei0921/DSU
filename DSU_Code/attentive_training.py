import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
from skimage import io
import os



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive



def loss_weight(input, target):

    _,c,w,h = target.size()
    loss_w = F.binary_cross_entropy_with_logits(input.clone().detach(), target.clone().detach(), reduction='none')

    loss_sample_tensor = torch.tensor(loss_w.data).mean(dim=1).mean(dim=1).mean(dim=1)
    loss_sample_weight = torch.softmax(1- loss_sample_tensor,dim=0)

    weight = loss_sample_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,c,w,h)

    return weight,loss_sample_weight



def update_pseudoLabel(l_weight, ppath, Sal_d,Non_Sal_d,Pred,epoch_num):
    batch_length = len(l_weight)
    update_num = int(batch_length*1.0)

    update_idex = np.argsort(-np.array(l_weight.cpu()))[:update_num]

    [images_path, gts_path] = ppath

    update_image = [images_path[i] for i in update_idex]
    update_gt = [gts_path[i] for i in update_idex]
    update_Sal = [Sal_d[i] for i in update_idex]
    update_non_Sal = [Non_Sal_d[i] for i in update_idex]
    update_Saliency = [Pred[i] for i in update_idex]


    # Backup pseudo label
    if not os.path.exists('./update'):
        os.mkdir('./update')
    dirname = './update/{}'.format(str(epoch_num))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.system("cp -r {} {}".format(str(gts_path[0][:-len(gts_path[0].split('/')[-1])]), dirname))


    # Update fake label
    if not os.path.exists('./update/temp'):
        os.mkdir('./update/temp')
    for i in range(len(update_idex)):
        res = update_Saliency[i] + update_Sal[i] - update_non_Sal[i]
        zero = torch.zeros_like(res)
        res = torch.tensor(torch.where(res <= 0.0, zero, res))
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        Sal_name = './update/temp/' + update_gt[i].split('/')[-1]
        io.imsave(Sal_name, np.uint8(res * 255))
        os.system('python ../DenseCRF/examples/dense_hsal.py {} {} {}'.format(update_image[i], Sal_name, update_gt[i]))
    return None

