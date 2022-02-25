"""
Example importable module holding customized ingredients of a workflow with hover.
Specifically for 3-channel image data in URLs.
"""

import os
import re
import numpy as np
import wrappy
import requests
from PIL import Image
from io import BytesIO


DIR_PATH = os.path.dirname(__file__)
RAW_CACHE_PATH = os.path.join(DIR_PATH, "raws.pkl")
IMG_CACHE_PATH = os.path.join(DIR_PATH, "imgs.pkl")
VEC_CACHE_PATH = os.path.join(DIR_PATH, "vecs.pkl")


@wrappy.memoize(
    cache_limit=50000,
    return_copy=False,
    persist_path=RAW_CACHE_PATH,
    persist_batch_size=100,
)
def url_to_content(url):
    """
    Turn a URL to response content.
    """
    response = requests.get(url)
    return response.content


@wrappy.memoize(
    cache_limit=50000,
    return_copy=False,
    persist_path=IMG_CACHE_PATH,
    persist_batch_size=100,
)
def url_to_image(url):
    """
    Turn a URL to a PIL Image.
    """
    img = Image.open(BytesIO(url_to_content(url))).convert("RGB")
    return img


def get_vectorizer():
    import torch
    from efficientnet_pytorch import EfficientNet
    from torchvision import transforms

    # EfficientNet is a series of pre-trained models
    # https://github.com/lukemelas/EfficientNet-PyTorch
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model.eval()

    # standard transformations for ImageNet-trained models
    tfms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # memoization can be useful if the function takes a while to run, which is common for images
    @wrappy.memoize(
        cache_limit=50000,
        return_copy=False,
        persist_path=VEC_CACHE_PATH,
        persist_batch_size=100,
    )
    def vectorizer(url):
        """
        Using logits on ImageNet-1000 classes.
        """
        img = tfms(url_to_image(url)).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)

        return outputs.detach().numpy().flatten()

    return vectorizer


def get_architecture():
    from hover.utils.common_nn import LogisticRegression

    return LogisticRegression


def get_state_dict_path():
    return os.path.join(DIR_PATH, "model.pt")
