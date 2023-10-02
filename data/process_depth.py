import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')


class DepthNorm(nn.Module):
    def __init__(
        self,
        max_depth=0,
        min_depth=0.01,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.scale = 1000.0  # nyuv2 abs.depth

    def forward(self, image):
        # image = np.array(image)
        depth_img = image / self.scale  # (H, W)   in meters
        depth_img = depth_img.clip(min=self.min_depth)
        if self.max_depth != 0:
            depth_img = depth_img.clip(max=self.max_depth)
            depth_img /= self.max_depth   #  0-1
        else:
            depth_img /= depth_img.max()
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())

def get_depth_transform(args):
    transform = transforms.Compose(
        [
            DepthNorm(max_depth=args.max_depth),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
            # transforms.Normalize((0.5, ), (0.5, ))  # 0-1 to norm distribution
            # transforms.Normalize((0.0418, ), (0.0295, ))  # sun rgb-d  imagebind
            # transforms.Normalize((0.02, ), (0.00295, ))  # nyuv2
        ]
    )
    return transform

def load_and_transform_depth(depth_path, transform):
    depth = opencv_loader(depth_path)
    depth_outputs = transform(depth)
    return {'pixel_values': depth_outputs}
