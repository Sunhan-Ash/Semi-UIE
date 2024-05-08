import torch
import torch.nn.functional as F


# Helper functions for channel adjustment and saturation
def adjust_channels_pytorch(r, ch1, ch2, mean1, mean2):
    loss = 1
    while loss > 0.01:
        J = mean1 - torch.mean(r)
        k = mean1 - torch.mean(ch2)
        r += J * ch1
        ch2 += k * ch1
        loss = torch.min(J, k)

def adjust_saturation_pytorch(im_org, satLevel1, satLevel2):
    imRGB = torch.zeros_like(im_org,device=im_org.device)
    for ch in range(3):
        q = torch.tensor([satLevel1[ch], 1 - satLevel2[ch]], device=im_org.device)
        tiles = torch.quantile(im_org[:, ch, :, :].flatten(), q).to(im_org.device)
        temp = im_org[:, ch, :, :]
        temp = torch.clamp(temp, tiles[0], tiles[1])
        bottom = temp.min()
        top = temp.max()
        imRGB[:, ch, :, :] = (temp - bottom) / (top - bottom)
    return imRGB

# def channel_difference_pytorch(channel):
#     kernel_size = 3
#     sigma = 1.4
#     # Create a Gaussian kernel
#     kernel = gaussian_kernel(kernel_size, sigma)
#     kernel = kernel.expand(channel.shape[1], 1, kernel_size, kernel_size)
#     filtered = F.conv2d(channel.unsqueeze(1), kernel, padding=kernel_size//2)
#     filtered = filtered.squeeze(1)
#     return channel - filtered
def gaussian_kernel_2d(kernel_size, sigma):
    """Generates a 2D Gaussian kernel."""
    ax = torch.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    # Normalize the kernel
    kernel = kernel / torch.sum(kernel)
    return kernel

def channel_difference_pytorch(channel):
    kernel_size = 3
    sigma = 1.4
    # Create a 2D Gaussian kernel
    kernel = gaussian_kernel_2d(kernel_size, sigma).to(channel.device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channel.size(0), 1, 1, 1)

    # Apply convolution with the Gaussian kernel
    filtered = F.conv2d(channel.unsqueeze(1), kernel, padding=kernel_size//2)
    
    # Select the first channel from the filtered result to match the input shape
    filtered = filtered[:, 0, :, :]

    return channel - filtered




def gaussian_kernel(kernel_size, sigma):
    # Generate a Gaussian kernel
    coords = torch.arange(kernel_size).to("cuda:0").float()
    coords -= (kernel_size - 1) / 2
    g = coords ** 2
    g = (-g / (2 * sigma ** 2)).exp()
    g /= g.sum()
    return g

# Example usage
# Assuming input_tensor is a PyTorch tensor with shape [4, 3, 256, 256]
# Move the tensor to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def LACC_pytorch_optimized(input_tensor):
    # Process each color channel
    Ar = 1 - torch.pow(input_tensor[:, 0, :, :], 1.2)
    meanAr = torch.mean(Ar)
    Ag = 1 - torch.pow(input_tensor[:, 1, :, :], 1.2)
    meanAg = torch.mean(Ag)
    Ab = 1 - torch.pow(input_tensor[:, 2, :, :], 1.2)
    meanAb = torch.mean(Ab)

    # Choose the channel with the maximum average value
    Am = torch.where(meanAr > meanAg, Ar, torch.where(meanAb > meanAr, Ab, Ag))

    # Re-extract the original channels
    r, g, b = [input_tensor[:, i, :, :] for i in range(3)]
    meanR, meanG, meanB = [torch.mean(ch) for ch in [r, g, b]]

    # Adjust red and blue channels based on green and blue channel means
    adjust_channels_pytorch(r, g, b, meanG, meanB)

    # Combine adjusted channels into a new image
    im_org = torch.stack([r, g, b], dim=1)

    # Saturation control
    R, G, B = [torch.sum(im_org[:, i, :, :]) for i in range(3)]
    Max = torch.max(torch.tensor([R, G, B]))
    ratio = Max / torch.tensor([R, G, B])

    satLevel1 = 0.002 * ratio
    satLevel2 = 0.002 * ratio

    # Adjust saturation for each channel
    imRGB = adjust_saturation_pytorch(im_org, satLevel1.to("cuda:0"), satLevel2.to("cuda:0"))

    # Apply Gaussian filter and calculate difference images
    DR, DG, DB = [channel_difference_pytorch(imRGB[:, i, :, :]) for i in range(3)]

    # Combine original, adjusted, and difference images
    CC = torch.zeros_like(imRGB)
    for i in range(3):
        temp1 = Am * imRGB[:, i, :, :]
        temp2 = (1 - Am) * input_tensor[:, i, :, :]
        temp3 = [DR, DG, DB][i]
        CC[:, i, :, :] = temp1 + temp2 + temp3
    return CC
    # Normalize the final output to be in the range of 0 to 1
    # normalized_result = torch.clamp(CC, 0, 1)

    # return normalized_result

# Re-attempting to run the optimized function
# Using a smaller tensor for demonstration, to reduce memory usage
# smaller_input_tensor = torch.rand(2, 3, 128, 128).to(device)  # Random tensor for demonstration

# # Apply the LACC function
# optimized_results = LACC_pytorch_optimized(smaller_input_tensor)




