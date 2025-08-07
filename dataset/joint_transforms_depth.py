import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, depth, mask, manual_random=None):
        assert img.size == mask.size
        assert img.size == depth.size
        for t in self.transforms:
            # print(type(depth))
            img, depth, mask = t(img, depth, mask, manual_random)
        return img, depth, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, depth, mask, manual_random=None):
        # print(1)
        if manual_random is None:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, depth, mask
        else:
            if manual_random < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, depth, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, depth, mask, manual_random=None):
        # print(2)
        assert img.size == mask.size
        assert img.size == depth.size
        return img.resize(self.size, Image.BILINEAR), depth.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)
