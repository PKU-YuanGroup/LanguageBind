from PIL import Image

from open_clip import image_transform, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


def image_loader(path):
    return Image.open(path)

def get_image_transform(args):
    preprocess_val = image_transform(
        args.image_size,
        is_train=False,
        mean=OPENAI_DATASET_MEAN,
        std=OPENAI_DATASET_STD,
    )
    return preprocess_val

def load_and_transform_image(
    image_path,
    transform,
):
    image = image_loader(image_path)
    image_outputs = transform(image)

    return {'pixel_values': image_outputs}