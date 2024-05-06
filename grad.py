import cv2
import numpy as np
import torch
import torch.fft as fft

def high_pass_filter_fft(image, threshold):
    # 将图像转换为频域表示
    image_fft = fft.fft2(image)

    # 创建高通滤波器
    height, width = image.shape[-2], image.shape[-1]
    center_h, center_w = height // 2, width // 2
    radius = int(threshold * min(center_h, center_w))
    mask = torch.ones((height, width))
    mask[center_h-radius:center_h+radius, center_w-radius:center_w+radius] = 0

    # 应用高通滤波器
    image_fft = image_fft * mask

    # 将结果转换回空间域
    filtered_image = torch.abs(fft.ifft2(image_fft))

    return filtered_image

def gradient_extraction(input_image, high_pass_threshold, canny_low_threshold, canny_high_threshold):
    # 初始化一个空的梯度图
    gradients = torch.zeros_like(input_image)

    # 对每个通道进行处理
    for i in range(input_image.shape[1]):
        channel_image = input_image[:, i:i+1]  # 提取单通道图像
        enhanced_image = high_pass_filter_fft(channel_image, high_pass_threshold)

        # 使用Canny边缘检测
        edges = cv2.Canny((enhanced_image.squeeze().cpu().numpy() * 255).astype(np.uint8), canny_low_threshold, canny_high_threshold)

        # 将边缘图像转换为PyTorch张量
        edges = torch.from_numpy(edges).unsqueeze(0).float()

        # 将处理后的边缘信息保存到对应的通道
        gradients[:, i:i+1] = edges

    return gradients

# 使用示例
input_image = torch.randn(1, 3, 256, 256)  # 输入为3通道的256x256图像
high_pass_threshold = 0.1  # FFT高通滤波的阈值，可以根据需要调整
canny_low_threshold = 50   # Canny边缘检测的低阈值
canny_high_threshold = 150  # Canny边缘检测的高阈值

edges = gradient_extraction(input_image, high_pass_threshold, canny_low_threshold, canny_high_threshold)
print(edges.shape)
