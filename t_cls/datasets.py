import cv2
import torch

from data.build_datasets import DataInfo
from data.process_thermal import get_thermal_transform
from torchvision import datasets

def get_thermal_dataset(args):
    data_path = args.thermal_data_path
    transform = get_thermal_transform(args)
    dataset = datasets.ImageFolder(data_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
