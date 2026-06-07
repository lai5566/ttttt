# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/image_pool.py
# Image pool for storing history of generated images (used in CycleGAN)

import random
import torch


class ImagePool:
    """Image buffer that stores previously generated images

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators. This technique helps stabilize
    GAN training by preventing mode collapse and oscillations.

    The buffer stores up to pool_size images. When the buffer is full:
    - 50% chance: return an image from the buffer and replace it with the new image
    - 50% chance: return the new image directly without storing

    Original paper: "Learning from Simulated and Unsupervised Images through Adversarial Training"
    https://arxiv.org/abs/1612.07828

    Args:
        pool_size (int): size of image buffer. If pool_size=0, no buffer is created
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool

        Args:
            images: latest generated images from the generator

        Returns:
            images from the buffer

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:
            # If pool size is 0, directly return input without buffering
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)

            if self.num_imgs < self.pool_size:
                # Buffer is not full yet, so store this image and return it
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # Buffer is full
                p = random.uniform(0, 1)
                if p > 0.5:
                    # 50% chance: return a random image from pool and replace it with current
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # 50% chance: just return current image without storing
                    return_images.append(image)

        # Concatenate all returned images into a batch
        return_images = torch.cat(return_images, 0)
        return return_images
