"""
Code reference: https://github.com/Mhaiyang/CVPR2021_PFNet
"""
import random
import numpy as np
from PIL import Image, ImageEnhance

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, edge):
        assert img.size == mask.size
        for t in self.transforms:
            # print(t)
            img, mask, edge = t(img, mask, edge)
        return img, mask, edge

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), edge.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, edge
    
#自己加的
class RandomVerticallyFlip(object):
    def __call__(self, img, mask, edge):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), edge.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask, edge
#随机剪裁
class randomCrop(object):
    def __call__(self, img, mask, edge):
        border = 30
        image_width = img.size[0]
        image_height = img.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return img.crop(random_region), mask.crop(random_region), edge.crop(random_region)

#随机旋转
class randomRotation(object):
    def __call__(self, img, mask, edge):    
        mode = Image.BICUBIC
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            img = img.rotate(random_angle, mode)
            mask = mask.rotate(random_angle, mode)    
            edge =  edge.rotate(random_angle, mode)         
        return img, mask, edge


class colorEnhance(object):
    def __call__(self, image, mask, edge):
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return image, mask, edge
#对mask做
class randomPeper(object):
    def __call__(self, img, mask, edge):
        mask = np.array(mask)
        noiseNum = int(0.001 * mask.shape[0] * mask.shape[1])

        edge = np.array(edge)
        noiseNum = int(0.001 * edge.shape[0] * edge.shape[1])
        for i in range(noiseNum):

            randX = random.randint(0, mask.shape[0] - 1)

            randY = random.randint(0, mask.shape[1] - 1)

            if random.randint(0, 1) == 0:
                mask[randX, randY] = 0
            else:
                mask[randX, randY] = 255
        return img,Image.fromarray(mask)


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask, edge):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.NEAREST)