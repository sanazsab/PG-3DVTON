#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

import numpy as np
from PIL import Image
from .pose_utils import heatmap_embedding
from torchvision import transforms
from torch.utils.data import DataLoader


def dewarped_image(model, result):
    with torch.no_grad():
        model.eval()
        # input
        cloth_image = result['cloth_image'].float().cuda()
        warped_cloth = result['warped_cloth'].cuda()
        

        grid, theta = model(warped_cloth, cloth_image)

        dewarped_cloth = F.grid_sample(warped_cloth, grid, padding_mode='border')
        
    return dewarped_cloth
