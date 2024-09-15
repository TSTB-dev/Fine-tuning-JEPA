from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms

logger = getLogger()

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')
    
    def get_color_distortion(s=1.0):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)    
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort
    
    transforms_list = []
    transforms_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transforms_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transforms_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transforms_list += [GaussianBlur(p=0.5)]
    transforms_list += [transforms.ToTensor()]
    transforms_list += [transforms.Normalize(normalization[0], normalization[1])]
    
    transform = transforms.Compose(transforms_list)
    return transform

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        
    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        
        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius.item()))