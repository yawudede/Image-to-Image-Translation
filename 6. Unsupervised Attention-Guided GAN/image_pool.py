import random
import torch


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []

        for image in images:
            image = torch.unsqueeze(image.data, dim=0)

            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images


class ImageMaskPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            self.masks = []

    def query(self, images, masks):
        if self.pool_size == 0:
            return images, masks

        return_images = []
        return_masks = []

        for image, mask in zip(images, masks):
            image = torch.unsqueeze(image.data, dim=0)
            mask = torch.unsqueeze(mask.data, dim=0)

            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                self.masks.append(mask)
                return_images.append(image)
                return_masks.append(mask)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp_image = self.images[random_id].clone()
                    tmp_mask = self.masks[random_id].clone()
                    self.images[random_id] = image
                    self.masks[random_id] = mask
                    return_images.append(tmp_image)
                    return_masks.append(tmp_mask)
                else:
                    return_images.append(image)
                    return_masks.append(mask)

        return_images = torch.cat(return_images, 0)
        return_masks = torch.cat(return_masks, 0)
        return return_images, return_masks