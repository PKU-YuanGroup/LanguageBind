import torch

from data.build_datasets import DataInfo
from open_clip import image_transform, OPENAI_DATASET_STD, OPENAI_DATASET_MEAN, get_tokenizer
from torchvision import datasets


def get_imagenet(args, split):
    assert split in ["val", "v2"]
    preprocess_val = image_transform(
        args.image_size,
        is_train=False,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
    )
    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        data_path = args.imagenet_val
        assert data_path
        dataset = datasets.ImageFolder(data_path, transform=preprocess_val)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
    )

    return DataInfo(dataloader=dataloader, sampler=None)
