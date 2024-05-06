import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from UIQM import UIQM
from torch.autograd import Variable
from UCIQE import uciqe_on_gpu as UCIQE
import pyiqa
import torch.utils.data as data
from dataset_all import EvalData
from utils import AverageMeter
import os
import numpy as np

iqa_musiq = pyiqa.create_metric('musiq',device = 'cpu')
iqa_uranker = pyiqa.create_metric('uranker',device = 'cpu')
# iqa_niqe = pyiqa.create_metric('niqe').cuda()
# iqa_psnr = pyiqa.create_metric('psnr').cuda()
# iqa_ssim = pyiqa.create_metric('ssim').cuda()

# Define the paths to the directories
enhanced_images_path = "../../second/MySecond/result3.4/UIEBD"
# enhanced_images_path = "./data/UIEBD2/val/GT"
# gt_images_path = "/media/xusunhan/USB/EUVP/test_samples/GTr"
# gt_images_path = "/media/xusunhan/USB/LUSI/val/GT"
# gt_images_path = "H:\\LUSI\\val\\GT"
gt_images_path ="./data/EUVP/test_samples/GT"
# gt_images_path ="H:\\UIEBD\\val\\GT"
# gt_images_path = "./data/UIEBD2/val/GT"
# Function to calculate SSIM and PSNR between two images

# Lists to store the results
ssim_values = []
psnr_values = []
musiq_values = []
uranker_values = []
niqe_values = []


# Iterate over the images in the directories
for filename in os.listdir(enhanced_images_path):
    enhanced_image_path = os.path.join(enhanced_images_path, filename)
    gt_image_path = os.path.join(gt_images_path, filename)
    musiq_value = iqa_musiq(enhanced_image_path).detach().item()
    uranker_value = iqa_uranker(enhanced_image_path).detach().item()
    # niqe_value = iqa_niqe(enhanced_image_path).detach().item()
    musiq_values.append(musiq_value)
    uranker_values.append(uranker_value)
    # niqe_values.append(niqe_value)
    # Check if the corresponding GT image exists
    # if os.path.exists(gt_image_path):
        # psnr_value = iqa_psnr(enhanced_image_path, gt_image_path).detach().item()
        # ssim_value = iqa_ssim(enhanced_image_path, gt_image_path).detach().item()
        # ssim_values.append(ssim_value)
        # psnr_values.append(psnr_value)


# Calculate average metrics
# average_ssim = np.mean(ssim_values) if ssim_values else 0
# average_psnr = np.mean(psnr_values) if psnr_values else 0
average_musiq = np.mean(musiq_values) if musiq_values else 0
average_uranker = np.mean(uranker_values) if uranker_values else 0
# average_niqe = np.mean(niqe_values) if niqe_values else 0

print(enhanced_image_path)
# print("Average SSIM:{:.4f}".format(average_ssim))
# print("Average PSNR:{:.4f}".format(average_psnr))
print("Average MUSIQ:{:.4f}".format(average_musiq))
print("Average uranker:{:.4f}".format(average_uranker))
# print("Average niqe:{:.4f}".format(average_niqe))




