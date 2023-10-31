import torch
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None


def _get_image_num_channels(img) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(img.ndim))

def _assert_channels(img, permitted) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError("Input image tensor permitted channel values are {}, but found {}".format(permitted, c))

def F_t_posterize(img, bits: int):

    # _assert_image_tensor(img)

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))
    if img.dtype != torch.uint8:
        raise TypeError("Only torch.uint8 image tensors are supported, but found {}".format(img.dtype))

    _assert_channels(img, [1, 3])
    mask = -int(2**(8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return img & mask


@torch.jit.unused
def _is_pil_image(img) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

@torch.jit.unused
def F_pil_posterize(img, bits):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    return ImageOps.posterize(img, bits)


def posterize(img, bits: int):
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        img (PIL Image or Tensor): Image to have its colors posterized.
            If img is torch Tensor, it should be of type torch.uint8 and
            it is expected to be in [..., 1 or 3, H, W] format, where ... means
            it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        bits (int): The number of bits to keep for each channel (0-8).
    Returns:
        PIL Image or Tensor: Posterized image.
    """
    if not (0 <= bits <= 8):
        raise ValueError('The number if bits should be between 0 and 8. Got {}'.format(bits))

    if not isinstance(img, torch.Tensor):
        return F_pil_posterize(img, bits)

    return F_t_posterize(img, bits)


class RandomPosterize(torch.nn.Module):
    """Posterize the image randomly with a given probability by reducing the
    number of bits for each color channel. If the image is torch Tensor, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being posterized. Default value is 0.5
    """

    def __init__(self, bits, p=0.5):
        super().__init__()
        # _log_api_usage_once(self)
        self.bits = bits
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be posterized.

        Returns:
            PIL Image or Tensor: Randomly posterized image.
        """
        if torch.rand(1).item() < self.p:
            return posterize(img, self.bits)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits},p={self.p})"