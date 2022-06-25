"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
import random

import numpy as np
import rp
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw
from torchvision import transforms
from rp import (
    as_float_image,
    as_numpy_array,
    as_rgb_image,
    get_image_dimensions,
    is_grayscale_image,
    is_image,
    load_image,
    DictReader,
)

def default_loader(path):
    #Unlike the original MUNIT implementation, this image loader
    #supports floating-point images, and can load .exr files
    return as_float_image(as_rgb_image(load_image(path)))


def get_image_files(folder):
    return [x for x in rp.get_all_files(folder, sort_by='number') if rp.is_image_file(x)]


def circle_mask(img, ox, oy, radius):
    #Takes in an image and 3 numbers
    #Outputs an image
    #
    #ox, oy is the top left corner of a circle
    #only the image img that circle is kept
    #all other pixels are turned black
    #
    #this is a pure function: it does not mutate inputs
    #example usage: https://pastebin.com/yTSmv1J0

    assert     is_image          (img)
    assert not is_grayscale_image(img)

    img=as_numpy_array(img).copy()
    img=as_float_image(img)

    height, width = get_image_dimensions(img)

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    x0 = width  * 0.5 - radius + ox
    x1 = width  * 0.5 + radius + ox
    y0 = height * 0.5 - radius + oy
    y1 = height * 0.5 + radius + oy
    
    draw.ellipse([x0, y0, x1, y1], fill=1)
    
    mask = as_numpy_array(mask)
    mask = as_rgb_image  (mask)
    mask = mask.astype(float)

    return img * mask


class ImageFolder(data.Dataset):

    def __init__(
        self,
        root,
        loader       = default_loader,
        return_paths = False,
        augmentation = {},
        precise      = False,
    ):
        imgs = get_image_files(root)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in: " + root + "\n")

        augmentation = DictReader(augmentation)

        self.root            = root
        self.precise         = precise
        self.imgs            = imgs
        self.return_paths    = return_paths
        self.loader          = loader
        self.output_size     = augmentation.output_size
        self.add_circle_mask = "circle_mask" in augmentation and augmentation.circle_mask == True
        self.rotate          = "rotate"      in augmentation and augmentation.rotate      == True
        self.contrast        = "contrast"    in augmentation and augmentation.contrast    == True

        self.skip_crop = False #When evaluating, we might set this to True...

        if "new_size_min" in augmentation and "new_size_max" in augmentation:
            self.new_size_min = augmentation.new_size_min
            self.new_size_max = augmentation.new_size_max
        else:
            self.new_size_min = min(self.output_size)
            self.new_size_max = min(self.output_size)

    def __getitem__(self, index):
        path = self.imgs[index]
        return self.process_image_path(path)

    def process_image_path(self, path):
        img = self.loader(path)

        # minOutputSize = min(self.output_size)
        # maxOutputSize = max(self.output_size)

        randAng  = random.random()*20-10
        randSize = random.randint(self.new_size_min, self.new_size_max)

        if self.add_circle_mask:
            height, width = get_image_dimensions(img)

            # minSize = min((width, height))
            maxSize = max((width, height))

            maxRadius = int(math.sqrt((width/2)**2 + (height/2)**2))
            minRadius = int(0.4*maxSize)
            
            maskRadius = random.randint( minRadius, maxRadius )

            maskOx = random.randint(int(-width  * 0.1), int(width  * 0.1))
            maskOy = random.randint(int(-height * 0.1), int(height * 0.1))

            img = circle_mask(img, maskOx, maskOy, maskRadius)

        assert isinstance(img, np.ndarray)

        img = img.astype(np.float32)
        img = transforms.functional.to_tensor(img)

        assert isinstance(img, torch.Tensor)

        if self.precise:
            #We don't want blurry boundaries on the UV maps
            interp = transforms.InterpolationMode.NEAREST
        else:
            interp = transforms.InterpolationMode.BILINEAR

        img = transforms.functional.resize(img, randSize, interp)

        if self.rotate:
            # this is disabled from get_all_data_loaders in utils.py
            img = transforms.functional.rotate(img, randAng, interp)
        
        C,H,W=img.shape

        ry = random.randint(0, max(H - self.output_size[1], 0))
        rx = random.randint(0, max(W - self.output_size[0], 0))

        if not self.skip_crop:
            img = transforms.functional.crop(img, ry, rx, self.output_size[1], self.output_size[0])

        if not self.precise:
            #We don't want to make UV maps brighter or dimmer, as this could cause arbitrary shifts in textures
            if self.contrast:
                # this is disabled from get_all_data_loaders in utils.py
                c = random.uniform( 0.75, 1.25)
                b = random.uniform(-0.10, 0.10)
                img = img * c + b

            
            # As it turns out, functional.normalize applies the same shift to any image regardless of its contents
            # The given mean and standard deviation are assumed to be the mean and standard deviation of the image
            # Then, with that assumption, it will apply a linear pixel-wise transform to turn those statustics
            # into the standard normal distribution: mean(0,0,0) std(0,0,0). 
            # Because it doesn't rely on the img itself, it means it's fully reversable without knowledge of the 
            # original input statistics.
            # TLDR: if I'm correct, the following line is equivalent to:
            #     img = img * 2 - 1
            img = transforms.functional.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
