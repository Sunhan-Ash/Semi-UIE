from tqdm import tqdm
import utils
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np
import os
import loss
import skimage.io as io
from datetime import datetime
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random
import cv2

class UIE_Runner():
    def __init__(self, options, type='train'):
        # manual_seed(options['seed'])

        self.type = type
        self.dataset_opt = options['dataset']
        train_folder = options['dataset']
        self.model_opt = options['model']
        self.training_opt = options['train']
        self.experiments_opt = options['experiments']
        self.test_opt = options['test']

        self.model = utils.build_model(self.model_opt)

        self.train_dataloader = utils.build_dataloader(self.dataset_opt, type='train')
        self.test_dataloader = utils.build_dataloader(self.dataset_opt, type='test')
        
        self.optimizer = utils.build_optimizer(self.training_opt, self.model)
        self.lr_scheduler = utils.build_lr_scheduler(self.training_opt, self.optimizer)
        
        self.tb_writer = SummaryWriter(os.path.join(self.experiments_opt['save_root'], 'tensorboard'))
        self.logger = utils.build_logger(self.experiments_opt)

    def main_loop(self):
        psnr_list = []
        ssim_list = []
        start_epoch = 0
        if self.model_opt['resume_ckpt_path']:
            ckpt = torch.load(self.model_opt['resume_ckpt_path'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            psnr_list.append(ckpt['max_psnr'])
            ssim_list.append(ckpt['max_ssim'])
            start_epoch = ckpt['epoch'] + 1
            for _ in range(start_epoch * 50):
                self.lr_scheduler.step()

        print(self.model)
        for epoch in range(start_epoch, self.training_opt['epoch']):
            print('================================ %s %d / %d ================================' % (self.experiments_opt['save_root'].split('/')[-1], epoch, self.training_opt['epoch']))
            loss = self.train_loop(epoch)
            torch.cuda.empty_cache()
            psnr, ssim = self.test_loop(epoch_num=epoch)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            self.logger.info(
                f"Epoch: {epoch}/{self.training_opt['epoch']}\t"
                f"Loss: {loss}\t"
                f"PSNR: {psnr} (max: {np.max(np.array(psnr_list))})\t"
                f"SSIM: {ssim} (max: {np.max(np.array(ssim_list))})\t"
            )
            if np.max(np.array(psnr_list)) == psnr or np.max(np.array(ssim_list)) == ssim:
                self.logger.warning(f"After {epoch+1} epochs trainingg, model achecieves best performance ==> PSNR: {psnr}, SSIM: {ssim}\n")
                # if epoch > 50:
                self.save(epoch, psnr, ssim)
            print()

    def main_test_loop(self):
        if self.test_opt['start_epoch'] >=0 and self.test_opt['end_epoch'] >=0 and self.test_opt['test_ckpt_path'] is None:
            for i in range(self.test_opt['start_epoch'], self.test_opt['end_epoch']):
                checkpoint_name = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'], f'checkpoint_{i}.pth')
                self.test_loop(checkpoint_name, i)
        else:
            self.test_loop(self.test_opt['test_ckpt_path'])
        
    def train_loop(self, epoch_num):
        total_loss = 0
        ranker_model = utils.build_model(self.training_opt['ranker_args']) if self.training_opt['loss_rank'] else None

        with tqdm(total=len(self.train_dataloader)) as t_bar:
            for iter_num, data in enumerate(self.train_dataloader):
                # put data to cuda device
                if self.model_opt['cuda']:
                    data = {key:value.cuda() for key, value in data.items()}
                
                # model prediction
                result = self.model(**data)
                
                # # loss and bp
                loss = self.build_loss(result, data['gt_img'], ranker_model)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # # print log
                # self.print()
                total_loss = (total_loss * iter_num + loss) / (iter_num + 1)
                t_bar.set_description('Epoch:%d/%d, loss:%.6f' % (epoch_num, self.training_opt['epoch'], total_loss))
                t_bar.update(1)

        
        self.tb_writer.add_scalar('train/loss', total_loss, epoch_num + 1)
        if self.training_opt['loss_rank']:
            ranker_model=ranker_model.cpu()
        return total_loss
            
    def test_loop(self, checkpoint_path=None, epoch_num=-1):
        if checkpoint_path == None:
            raise NotImplementedError('checkpoint_name can not be NoneType!')

        with torch.no_grad():
            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()
            if checkpoint_path:
                ckpt_dict = torch.load(checkpoint_path)['net']
                self.model.load_state_dict(ckpt_dict)
            if self.test_opt['save_image']:
                save_root = os.path.join(self.experiments_opt['save_root'], 'results')
                utils.make_dir(save_root)

            with tqdm(total=len(self.test_dataloader)) as t_bar:
                for iter_num, data in enumerate(self.test_dataloader):
                    _, _, h, w = data['gt_img'].shape
                    gt_img = data['gt_img'][0].permute(1, 2, 0).detach().numpy()
                    if self.model_opt['cuda']:
                        data = {key:value.cuda() for key, value in data.items()}

                    upsample = nn.UpsamplingBilinear2d((h, w))
                    pred_img = upsample(utils.normalize_img(self.model(**data)))
                    pred_img = pred_img[0].permute(1, 2, 0).detach().cpu().numpy()
                    if self.test_opt['save_image']:
                        cv2.imwrite(os.path.join(self.experiments_opt['save_root'], 'results', str(iter_num)+'.png'), pred_img[:, :, ::-1] * 255.0)

                    psnr = utils.calc_psnr(pred_img, gt_img, is_for_torch=False)
                    ssim = utils.calc_ssim(pred_img, gt_img, is_for_torch=False)

                    psnr_meter.update(psnr)
                    ssim_meter.update(ssim)

                    # update bar
                    if checkpoint_path:
                        t_bar.set_description('checkpoint: %s, psnr:%.6f, ssim:%.6f' % (checkpoint_path.split('/')[-1], psnr_meter.avg, ssim_meter.avg))
                    if epoch_num >= 0:
                        t_bar.set_description('Epoch:%d/%d, psnr:%.6f, ssim:%.6f' % (epoch_num, self.training_opt['epoch'], psnr_meter.avg, ssim_meter.avg))

                    t_bar.update(1)
        if epoch_num >= 0:
            self.tb_writer.add_scalar('valid/psnr', psnr_meter.avg, epoch_num + 1)
            self.tb_writer.add_scalar('valid/ssim', ssim_meter.avg, epoch_num + 1)

        return psnr_meter.avg, ssim_meter.avg
            
    
    def save(self, epoch_num, psnr, ssim):
        # path for saving
        path = os.path.join(self.experiments_opt['save_root'], self.experiments_opt['checkpoints'])
        utils.make_dir(path)
            
        checkpoint = {
        "net": self.model.state_dict(),
        'optimizer':self.optimizer.state_dict(),
        "epoch": epoch_num,
        "max_psnr": psnr,
        "max_ssim": ssim
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_{epoch_num}.pth'))
    
    def build_loss(self, pred, gt, ranker_model):
        loss_total = 0
        Loss_L1 = nn.L1Loss().cuda()
        loss_total = loss_total + self.training_opt['loss_coff'][0] * Loss_L1(pred, gt)

        if self.training_opt['loss_vgg']:
            Loss_VGG = loss.make_perception_loss(self.training_opt.get('loss_vgg_args')).cuda()
            loss_total = loss_total + self.training_opt['loss_coff'][1] * Loss_VGG(pred, gt)
            del Loss_VGG
        if self.training_opt['loss_rank']:
            loss_total = loss_total + self.training_opt['loss_coff'][2] * loss.ranker_loss(ranker_model, pred)

        return loss_total
