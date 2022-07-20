import torch
import torch.nn.functional as F

def downsample(image, scale_factor):    
    image = F.avg_pool2d(torch.from_numpy(image).unsqueeze(0).unsqueeze(0).double(), scale_factor)      
    image = image.squeeze().numpy()       
    return image
 
def bicubic(image, target_size):       
    image = F.interpolate(torch.from_numpy(image).unsqueeze(0).unsqueeze(0).double(), target_size, mode='bicubic', align_corners=False)                 
    image = image.squeeze().numpy()
    return image
