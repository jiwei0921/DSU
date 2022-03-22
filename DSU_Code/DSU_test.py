import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from skimage import io
from tqdm import trange
# from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from model.Sal_CNN import Sal_CNN
from evaluateSOD.main import evalateSOD
from data import test_dataset



def eval_data(dataset_path, test_datasets, ckpt_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--snapshot', type=str, default=ckpt_name, help='checkpoint name')
    cfg = parser.parse_args()


    model_rgb = CPD_ResNet()
    model_depth = CPD_ResNet()
    model = Sal_CNN()
    model_rgb.load_state_dict(torch.load('./ckpt/'+'DSU_rgb.pth' +cfg.snapshot))
    model_depth.load_state_dict(torch.load('./ckpt/' + 'DSU_depth.pth' +cfg.snapshot))
    model.load_state_dict(torch.load('./ckpt/' +'DSU.pth' + cfg.snapshot))



    cuda = torch.cuda.is_available()
    if cuda:
        model_rgb.cuda()
        model_depth.cuda()
        model.cuda()
    model_rgb.eval()
    model_depth.eval()
    model.eval()



    for dataset in test_datasets:
        save_path = './results/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/test_images/'
        gt_root = dataset_path + dataset + '/test_masks/'
        depth_root = dataset_path + dataset + '/test_depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, cfg.testsize)
        print('Evaluating dataset: %s' %(dataset))


        '''~~~ Our DSU FRAMEWORK~~~'''
        for i in trange(test_loader.size):
            image, gt, depth, name = test_loader.load_data()

            if cuda:
                image = image.cuda()
                depth = depth.cuda()

            # The inference stage involves only RGB stream, i.e., only RGB image is used for predicting.
            _, res_r, _ = model_rgb(image)

            res = res_r
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            io.imsave(save_path+name, np.uint8(res * 255))
        _ = evalateSOD(save_path, gt_root, dataset,ckpt_name,switch=False) # This eval code is for reference.
        # All reported results in this paper are evaluated by the same code
        # as in 'https://github.com/jiwei0921/Saliency-Evaluation-Toolbox'.
    return



if __name__ == '__main__':
    dataset_path = '../Dataset/test_data/'
    #test_datasets=['NLPR']
    test_datasets = ['DUT','NJUD', 'NLPR','STERE1000', 'SIP','LFSD', 'RGBD135','SSD']

    ckpt_name = '.24'
    eval_data(dataset_path,test_datasets,ckpt_name)
