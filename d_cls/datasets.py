import cv2
import torch

from data.build_datasets import DataInfo
from data.process_depth import get_depth_transform, opencv_loader
from torchvision import datasets

def get_depth_dataset(args):
    data_path = args.depth_data_path
    transform = get_depth_transform(args)
    dataset = datasets.ImageFolder(data_path, transform=transform, loader=opencv_loader)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
