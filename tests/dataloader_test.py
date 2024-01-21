import unittest

import matplotlib.pyplot as plt
import torch

from utils.aos_loader import _get_dataloader


class TestDataLoader(unittest.TestCase):
    def test_train_loader(self):
        data_loader, _data = _get_dataloader("../data/test", focal_heights=("0", "0"), image_resolution=(128, 128),
                                         batch_size=1)
        for (x, y) in data_loader:
            self.assertEqual(type(x), torch.Tensor)
            self.assertEqual(type(y), torch.Tensor)

        transformed_images = []
        transformed_images_y = []
        for i in range(8):
            _data.update_epoch(i)
            transformed_images.append(_data[0][0][0])
            transformed_images.append(_data[0][1][0])

        self.display_images_side_by_side(*transformed_images)
        self.display_images_side_by_side(*transformed_images_y)

    def display_images_side_by_side(self, *images):
        # Create sub-plots with larger figure size
        fig, axes = plt.subplots(1, len(images), figsize=(10 * len(images), 10))  # Adjust as needed

        # If there's just one image, axes will not be a list
        if len(images) == 1:
            axes = [axes]

        # Display each image in a subplot
        for ax, img in zip(axes, images):
            ax.imshow(img, cmap='gray')  # 'gray' colormap for black-and-white image
            ax.axis('off')

        plt.show()