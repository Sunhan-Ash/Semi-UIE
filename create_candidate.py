import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import transforms
import cv2

# 初始化可靠的数据集

# 输入目录
input_dir = 'data/UIEBD1/unlabeled/input'

# 结果目录
result_dir = 'data/UIEBD1/unlabeled/candidate'

# 获取输入目录下的所有文件路径
input_lists = glob(join(input_dir, '*.*'))

# 遍历输入目录中的每个文件
for gen_path in zip(input_lists):
    # 创建一个大小为(3, 256, 256)的全零张量
    img = torch.zeros((3, 256, 256))

    # 获取图片名称
    img_name = gen_path[0].split('/')[-1]

    # 打印图片名称
    print(img_name)

    # 创建一个transforms.ToPILImage()对象
    toPil = transforms.ToPILImage()

    # 将张量转换为PIL图像，再转换为RGB模式
    res = toPil(img).convert('RGB')

    # 检查结果目录是否存在，如果不存在则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 保存图像到结果目录中，使用图片名称作为文件名

    res.save(os.path.join(result_dir, img_name))
