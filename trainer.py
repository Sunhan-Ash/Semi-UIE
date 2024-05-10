import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import torchvision
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from adamp import AdamP
from torchvision.models import vgg16
from loss.losses import *
#from model import GetGradientNopadding
from loss.contrast import ContrastLoss
import pyiqa
from torch.nn.utils import clip_grad_norm_
#from LACC_pyotrch import LACC_pytorch_optimized as reLACC
@torch.no_grad()
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
    
class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 20
        self.best_psnr = 0
        self.NR = 0
        self.best_eval = 0
        self.loss_unsup = nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_grad = nn.L1Loss().cuda()
        self.loss_cr = ContrastLoss().cuda()
        self.loss_uranker = ranker_loss().cuda()
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.iqa_metric = pyiqa.create_metric('musiq', as_loss=True).cuda()
        self.iqa_uranker = pyiqa.create_metric('uranker', as_loss=True).cuda()
        self.iqa_niqe = pyiqa.create_metric('niqe', as_loss=True).cuda()
        self.iqa_psnr = pyiqa.create_metric('psnr', as_loss=True).cuda()
        self.iqa_ssim = pyiqa.create_metric('ssim', as_loss=True).cuda()
        
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.args = args
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.curiter = 0
        self.model.cuda()
        
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        # set optimizer and learning rate
        self.optimizer_s = AdamP(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        # self.lr_scheduler_s = lr_scheduler.StepLR(self.optimizer_s, step_size=100, gamma=0.1)
        # self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)
        self.lr_scheduler_s = lr_scheduler.CosineAnnealingLR(self.optimizer_s, T_max=args.num_epochs,eta_min =args.lr * 1e-2)
    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        # 计算EMA的平滑系数alpha,这里采用指数衰减策略
        # 遍历teacher和student模型的参数
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            # 对teacher的参数进行EMA更新,采用momentum形式,保持teacher参数不变化
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image, image_l):
        with torch.no_grad():
            # 将image和image_l作为输入, 经过teacher模型Forward, 得到预测结果predict_target_ul。
            # 最后返回预测结果。
            # 这样可以用teacher模型进行预测,但不会更新teacher的参数。避免了 teacher参数随着student变化。
            predict_target_ul = self.tmodel(image, image_l)

        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False
    
    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, score_r):
        N = teacher_predict.shape[0]  # 获取样本数量
        score_t = self.iqa_metric(teacher_predict).detach().cpu().numpy()  # 计算教师模型预测结果的 IQA 得分
        score_s = self.iqa_metric(student_predict).detach().cpu().numpy()  # 计算学生模型预测结果的 IQA 得分
        positive_sample = positive_list.clone()  # 克隆一个正样本列表

        for idx in range(0, N):  # 遍历每个样本
            if score_t > score_s:  # 如果教师模型的预测得分大于学生模型的预测得分
                if score_t > score_r:  # 如果教师模型的预测得分大于某个阈值 score_r
                    # print("reliable bank 已更新，当前score：{:.5f}".format(score_t))
                    positive_sample[idx] = teacher_predict[idx]  # 将该样本设为可靠样本
                    # 更新 "reliable bank"
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))  # 将通道维度移到最后
                    temp_c = np.clip(temp_c, 0, 1)  # 将像素值限制在 [0, 1] 范围内
                    arr_c = (temp_c * 255).astype(np.uint8)  # 将像素值映射到 [0, 255] 范围并转为无符号整数
                    arr_c = Image.fromarray(arr_c)  # 创建一个图像对象
                    arr_c.save('%s' % p_name[idx])  # 将图像保存到指定路径

        # 释放变量占用的内存
        del N, score_r, score_s, score_t, teacher_predict, student_predict, positive_list
        return positive_sample  # 返回更新后的正样本列表
    
    # def reLACC(self,x):
    #     with torch.no_grad():
    #         x = LACC(x)
    #         x = normalize_img(x)
    #     return x
    def train(self):
        early_stop_epoch = self.args.early_stop_epoch
        best_psnr = self.best_psnr  # 初始化最佳psnr为0
        best_NR = self.NR
        best_eval = self.best_eval
        
        self.freeze_teachers_parameters()  # 冻结教师模型的参数
        # if self.start_epoch == 1:
        if self.start_epoch == 1 and  self.args.resume == 'False':
            initialize_weights(self.model)  # 如果是第一个 epoch，初始化模型参数
        else:
            checkpoint = torch.load(self.args.resume_path)  # 否则，从预训练模型中加载状态字典
            self.model.load_state_dict(checkpoint['state_dict'])  # 加载模型的状态字典
            self.start_epoch = checkpoint['epoch']
            # self.start_epoch = 0
            best_psnr = checkpoint['best_psnr']
            best_MUSIQ = checkpoint['best_MUSIQ']
            best_NR = checkpoint['best_MUSIQ']+checkpoint['best_URanker']
            best_eval = checkpoint['best_MUSIQ']+checkpoint['best_URanker']+checkpoint['best_psnr']
            # best_eval = checkpoint['eval']
        for epoch in range(self.start_epoch, self.epochs + 1):  # 开始迭代训练
            loss_ave, psnr_train = self._train_epoch(epoch)  # 调用 _train_epoch 方法进行训练，并获取平均损失和 PSNR 值
            loss_val = loss_ave.item() / self.args.crop_size * self.args.train_batchsize  # 计算验证集上的损失值
            train_psnr = sum(psnr_train) / len(psnr_train)  # 计算训练集上的平均 PSNR 值
            psnr_val, MUSIQ, ssim_val, val_niqe, val_uranker = self._valid_epoch(max(0, epoch))  # 调用 _valid_epoch 方法计算验证集上的 PSNR 值
            # val_psnr = sum(psnr_val) / len(psnr_val)  # 计算验证集上的平均 PSNR 值

            # 打印当前 epoch 的统计信息，包括损失和 PSNR 值
            print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, val MUSIQ :%.4f,val SSIM :%.4f,val NIQE:%.4f,val URanker:%.4f,lr: %.8f' % (
                epoch, loss_val, train_psnr, psnr_val, MUSIQ, ssim_val, val_niqe, val_uranker, self.lr_scheduler_s.get_last_lr()[0]))

            eval = psnr_val + MUSIQ + ssim_val - val_niqe + val_uranker
            NR = MUSIQ + val_uranker
            # eval = val_uranker
            if eval > best_eval:
                # early_stop_epoch = self.args.early_stop_epoch
                best_eval = eval
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict(),
                         'best_psnr': psnr_val,
                         'best_MUSIQ': MUSIQ,
                         'best_SSIM': ssim_val,
                         'best_NIQE':val_niqe,
                         'best_URanker':val_uranker}
                ckpt_name = str(self.args.save_path) + 'best_in_evaluation.pth'
                torch.save(state, ckpt_name)
                print("Saving a checkpoint with now psnr: {} ...".format(psnr_val))
                print("Saving a checkpoint with now MUSIQ: {} ...".format(MUSIQ))
                print("Saving a checkpoint with now SSIM: {} ...".format(ssim_val))
                print("Saving a checkpoint with now NIQE: {} ...".format(val_niqe))
                print("Saving a checkpoint with now URanker: {} ...".format(val_uranker))
                print("Saving a checkpoint with now eval: {} ...".format(best_eval))
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict(),
                         'best_psnr': psnr_val,
                         'best_MUSIQ': MUSIQ,
                         'best_SSIM': ssim_val,
                         'best_NIQE':val_niqe,
                         'best_URanker':val_uranker}
                ckpt_name = str(self.args.save_path) + 'best_in_psnr.pth'
                torch.save(state, ckpt_name)
                print("Saving a checkpoint with best psnr: {} ...".format(best_psnr))
                
            if NR > best_NR:
                best_NR = NR
                state = {'arch': type(self.model).__name__,
                         'epoch': epoch,
                         'state_dict': self.model.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict(),
                         'best_psnr': psnr_val,
                         'best_MUSIQ': MUSIQ,
                         'best_SSIM': ssim_val,
                         'best_NIQE':val_niqe,
                         'best_URanker':val_uranker}
                ckpt_name = str(self.args.save_path) + 'best_in_NR.pth'
                torch.save(state, ckpt_name)
                # print("Saving a checkpoint with best psnr: {} ...".format(best_psnr))
                print("Saving a checkpoint with now MUSIQ: {} ...".format(MUSIQ))
                print("Saving a checkpoint with now URanker: {} ...".format(val_uranker))
    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()  # 创建一个用于计算平均值的容器，用于监测监督损失
        unsup_loss = AverageMeter()  # 创建一个用于计算平均值的容器，用于监测无监督损失
        loss_total_ave = 0.0  # 初始化总损失的平均值
        psnr_train = []  # 创建一个空列表，用于存储训练过程中的 PSNR（峰值信噪比）值
        self.model.train()  # 设置模型为训练模式
        self.freeze_teachers_parameters()  # 冻结教师模型的参数
        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))  # 创建一个迭代器，同时遍历监督和无监督数据加载器
        tbar = range(len(self.unsupervised_loader))  # 创建一个范围对象，用于显示训练进度条
        tbar = tqdm(tbar, ncols=130, leave=True)  # 创建一个 tqdm 进度条对象，用于可视化训练进度
        for i in tbar:  # 开始迭代训练过程
            # 从数据加载器中获取数据
            (img_data, label, img_la), (unpaired_data_w, unpaired_data_s, unpaired_la, p_list, p_name) = next(
                train_loader)
            # 将数据移动到 GPU 上并封装为 Variable（在较新版本的 PyTorch 中通常使用 tensor 直接操作）
            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            img_la = Variable(img_la).cuda(non_blocking=True)
            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            unpaired_la = Variable(unpaired_la).cuda(non_blocking=True)
            p_list = Variable(p_list).cuda(non_blocking=True)
            # 教师模型的输出
            predict_target_u = self.predict_with_out_grad(unpaired_data_w, unpaired_la)
            origin_predict = predict_target_u.detach().clone()
            # 学生模型的输出
            outputs_l = self.model(img_data, img_la)
            outputs_ul = self.model(unpaired_data_s, unpaired_la)
            # 计算各种损失
            structure_loss = self.loss_str(outputs_l, label)
            perpetual_loss = self.loss_per(outputs_l, label)

            loss_sup = structure_loss + 0.3 * perpetual_loss #group1
            # loss_sup = structure_loss + 0.3 * perpetual_loss #group2
            # loss_sup = 0.7*structure_loss + 0.3 * perpetual_loss #group3
            
            # 更新监督损失的平均值
            sup_loss.update(loss_sup.mean().item())
            # 计算 IQA（图像质量评价）指标并将结果转移到 CPU 上
            score_r = self.iqa_metric(p_list).detach().cpu().numpy()
            # 根据一些条件获取可靠样本
            p_sample = self.get_reliable(predict_target_u, outputs_ul, p_list, p_name, score_r)
            # 计算无监督损失
            # loss_unsu = 0.3*self.loss_unsup(outputs_ul, p_sample) + 0.3*self.loss_cr(outputs_ul, p_sample, unpaired_data_s)+0.4*self.loss_uranker(outputs_ul) #group1
            # loss_unsu = 0.5*self.loss_unsup(outputs_ul, p_sample) + 0.5*self.loss_cr(outputs_ul, p_sample, unpaired_data_s)#group2
            # loss_unsu = 0.5*self.loss_unsup(outputs_ul, p_sample) + 0.5*self.loss_cr(outputs_ul, p_sample, unpaired_data_s)+self.loss_uranker(outputs_ul) #group3
            loss_unsu = self.loss_unsup(outputs_ul, p_sample) + self.loss_cr(outputs_ul, p_sample, unpaired_data_s)+self.loss_uranker(outputs_ul) 
            # loss_unsu = self.loss_unsup(outputs_ul, p_sample)+self.loss_uranker(outputs_ul) #group4
            # 更新无监督损失的平均值
            unsup_loss.update(loss_unsu.mean().item())
            # 获取当前的一致性权重
            consistency_weight = self.get_current_consistency_weight(epoch)
            # 计算总损失
            total_loss = consistency_weight * loss_unsu + (1-consistency_weight)*loss_sup
            total_loss = total_loss.mean()
            # 计算并存储 PSNR 值
            psnr_train.extend(to_psnr(outputs_l, label))
            # 梯度清零并反向传播计算梯度
            self.optimizer_s.zero_grad()
            total_loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=0.8) 
            self.optimizer_s.step()

            # 更新进度条的描述
            tbar.set_description(
                'Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}|'.format(epoch, sup_loss.avg, unsup_loss.avg))

            # 释放变量占用的 GPU 内存
            del img_data, label, unpaired_data_w, unpaired_data_s, img_la, unpaired_la,

            # 在不计算梯度的情况下更新教师模型
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        # 更新总平均损失
        loss_total_ave = loss_total_ave + total_loss

        # 将损失和 PSNR 值写入 TensorBoard
        self.writer.add_scalar('Train_loss', total_loss, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)

        # 更新学习率调度器
        # clip_grad_norm_(self.model.parameters(), max_norm=0.8) 
        self.lr_scheduler_s.step(epoch=epoch - 1)

        # 返回总平均损失和 PSNR 值
        return loss_total_ave, psnr_train

    def _valid_epoch(self, epoch):
        psnr_val = []  # 创建一个列表，用于存储 PSNR 值
        self.model.eval()  # 设置模型为评估模式
        self.tmodel.eval()  # 设置教师模型为评估模式
        val_psnr = AverageMeter()  # 创建一个用于计算平均值的容器，用于监测验证集的 PSNR 值
        val_ssim = AverageMeter()  # 创建一个用于计算平均值的容器，用于监测验证集的 SSIM 值
        val_iqa = AverageMeter()
        val_niqe = AverageMeter()
        val_uranker = AverageMeter()
        total_loss_val = AverageMeter()  # 创建一个用于计算平均值的容器，用于监测验证集的损失值
        tbar = tqdm(self.val_loader, ncols=130)  # 创建一个 tqdm 进度条对象，用于可视化验证进度
        with torch.no_grad():  # 在评估过程中不计算梯度
            for i, (val_data, val_label, val_la) in enumerate(tbar):  # 遍历验证集
                val_data = Variable(val_data).cuda()  # 将验证数据移动到 GPU 并封装为 Variable
                N = len(val_data)
                # val_data = reLACC(val_data).cuda()
                val_label = Variable(val_label).cuda()  # 将验证标签移动到 GPU 并封装为 Variable
                val_la = Variable(val_la).cuda()  # 将验证标签移动到 GPU 并封装为 Variable
                # 前向传播计算验证集上的输出
                val_output = self.model(val_data, val_la)
                # val_output = normalize_img(val_output)
                iqa = self.iqa_metric(val_output).detach()
                # 计算验证集上的 PSNR 和 SSIM 值
                temp_psnr = self.iqa_psnr(val_output, val_label)
                temp_ssim = self.iqa_ssim(val_output, val_label)
                # temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                # niqe = self.iqa_niqe(val_output).detach()
                uranker = self.iqa_uranker(val_output).detach()
                val_psnr.update(temp_psnr, N)  # 更新 PSNR 平均值容器
                val_ssim.update(temp_ssim, N)  # 更新 SSIM 平均值容器
                val_iqa.update(iqa, N)  # 更新 IQA平均值容器
                # val_niqe.update(niqe, N)
                val_niqe.update(0, N)
                val_uranker.update(uranker,N)
                # psnr_val.extend(to_psnr(val_output, val_label))  # 将 PSNR 值扩展到列表中
                tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}| MUSIQ:{:.4f}'.format(
                    "Eval-Student", epoch, val_psnr.avg, val_ssim.avg, val_iqa.avg))  # 更新进度条描述信息
                # tbar.set_description('{} Epoch {} | PSNR: {:.4f}| MUSIQ:{:.4f}'.format(
                #     "Eval-Student", epoch, val_psnr.avg, val_ssim.avg, val_iqa.avg))  # 更新进度条描述信息

            # 将验证结果写入 TensorBoard
            self.writer.add_scalar('Val_psnr', val_psnr.avg, global_step=epoch)
            # print('Val_psnr', val_psnr.avg)
            self.writer.add_scalar('Val_ssim', val_ssim.avg, global_step=epoch)
            self.writer.add_scalar('Val_MUSIQ', val_iqa.avg, global_step=epoch)
            # self.writer.add_scalar('Val_niqe', val_niqe.avg, global_step=epoch)
            self.writer.add_scalar('Val_URanker', val_uranker.avg, global_step=epoch)

            # 释放变量占用的 GPU 内存
            del val_output, val_label, val_data
            return val_psnr.avg, val_iqa.avg, val_ssim.avg,val_niqe.avg,val_uranker.avg # 返回 PSNR 值列表

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # 指数增加的 ramp-up
        if rampup_length == 0:
            return 1.0  # 如果 rampup_length 为 0，则直接返回 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)  # 将 current 值限制在 [0, rampup_length] 范围内
            phase = 1.0 - current / rampup_length  # 计算 phase，表示当前阶段在 ramp-up 过程中的位置
            return float(np.exp(-5.0 * phase * phase))  # 使用 sigmoid 曲线计算阶段的权重

