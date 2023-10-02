import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD



def get_thermal_transform(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
        ]
    )
    return transform

def load_and_transform_thermal(thermal_path, transform):
    thermal = Image.open(thermal_path)
    thermal_outputs = transform(thermal)
    return {'pixel_values': thermal_outputs}
