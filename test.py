import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from UIE14_agg4_no_prior import *
from PIL import Image
from adamp import AdamP
# my import
# from model import AIMnet
from dataset_all import TestData
# from LACC_pyotrch import LACC_pytorch_optimized as reLACC
import pyiqa
from LACC import LACC

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        # img: b x c x h x w
        b, c, h, w = img.shape
        # temp_img = img.view(b, c, h*w)
        temp_img = img.reshape(b, c, h*w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-6)
        
        # img = temp_img.view(b, c, h, w)
        img = temp_img.reshape(b, c, h, w)
    
    return img

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
iqa_metric = pyiqa.create_metric('musiq').cuda()
iqa_psnr = pyiqa.create_metric('psnr').cuda()
iqa_ssim = pyiqa.create_metric('ssim').cuda()
iqa_uranker = pyiqa.create_metric('uranker').cuda()
iqa_niqe = pyiqa.create_metric('niqe').cuda()

bz = 1
# model_root = './model/ckpt/best_p_model.pth'
# model_root = './model/ckpt/best_M_model.pth'
# model_root = './model/ckpt/best_p_model_24.63_56.41.pth'
model_root = './model/ckpt/best_in_NR.pth'
# model_root = './pretrained/model.pth'
# input_root = './data/UIEBD2/val'
input_root = './data/UIEBD2/val'
save_path = './result3.13/UIEBD'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)
Mydata_ = TestData(input_root)
data_load = data.DataLoader(Mydata_, batch_size=bz)

model = UIE_Second().cuda()
# model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0])
optimizer = AdamP(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_dict'])
# epoch = checkpoint['epoch']
model.eval()
print('START!')
N = len(data_load)
val_iqa = AverageMeter()
val_psnr = AverageMeter()
val_ssim = AverageMeter()
val_uranker = AverageMeter()
val_niqe = AverageMeter()
if 1:
    print('Load model successfully!')
    for data_idx, data_ in enumerate(data_load):
        data_input, data_GT ,data_la = data_

        data_input = Variable(data_input).cuda()
        data_la = Variable(data_la).cuda()
        # data_input = reLACC(data_input).cuda()
        # N = len(data_load)
        print(data_idx)
        with torch.no_grad():
            name = Mydata_.A_paths[data_idx].split('/')[-1]
            print(name)
            result= model(data_input, data_la)
            temp_res = result
            # temp_res = normalize_img(result)
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            iqa_scores = iqa_metric(temp_res)
            uranker_scores = iqa_uranker(temp_res)
            # niqe_scores = iqa_niqe(temp_res)
            psnr_scores = iqa_psnr(temp_res,data_GT)
            ssim_scores = iqa_ssim(temp_res,data_GT)
            
            val_iqa.update(iqa_scores, N)
            val_uranker.update(uranker_scores, N)
            # val_niqe.update(niqe_scores, N)
            val_psnr.update(psnr_scores, N)
            val_ssim.update(ssim_scores, N)
            temp_res = np.transpose(temp_res[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s/%s' % (save_path, name))
            print('result saved!')
print("the MUSIQ is {:.4f}, the Uranker is {:.4f}, the PSNR is {:.4f}, the SSIM is {:.4f}".format(
    val_iqa.avg.item(),    # Convert tensor to a Python number
    val_uranker.avg.item(),
    # val_niqe.avg.item(),
    val_psnr.avg.item(),
    val_ssim.avg.item()
))

print('finished!')
