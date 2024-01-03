import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_test_dataloader(focal_heights=("0",), image_resolution=(128, 128)):
    return _get_dataloader("./data/test", focal_heights, image_resolution)


def get_train_dataloader(focal_heights=("0",), image_resolution=(128, 128)):
    return _get_dataloader("./data/train", focal_heights, image_resolution)


def _get_dataloader(main_dir, image_resolution, focal_heights, **kwargs):
    """

    :param main_dir:
    :param transform:
    :param focal_heights: tuple of focal heights to use
    """
    transform = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.ToTensor(),
    ])

    data = AosDataset(main_dir, transform, focal_heights)

    return DataLoader(data, shuffle=True, **kwargs)


class AosDataset(Dataset):
    def __init__(self, main_dir, transform, focal_heights):
        self.main_dir = main_dir
        self.transform = transform
        self.focal_heights = focal_heights
        all_imgs = glob.glob(os.path.join(main_dir, '*.png'))
        self.index_map = get_index_map(all_imgs)
        pass

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        batch_index, sample_index = self.index_map[idx]
        image_names = [os.path.join(self.main_dir, f"{batch_index}_{sample_index}-aos_thermal-{fh}.png") for fh in self.focal_heights]
        gt_name = os.path.join(self.main_dir, f"{batch_index}_{sample_index}_GT_pose_0_thermal.png")

        images = [Image.open(img_name).convert("L") for img_name in image_names]
        gt_image = Image.open(gt_name).convert("L")

        if self.transform is not None:
            images = [self.transform(image) for image in images]
            gt_image = self.transform(gt_image)
            images = torch.cat(images)

        return images, gt_image


def extract_indices(filename):
    filename = os.path.basename(filename)
    # Split the filename on the hyphen ('-')
    filename_split = filename.split('-')
    # The batch index and sample index should now be the first element in the resulting list, separated by an underscore ('_')
    indices = filename_split[0].split('_')
    # Assuming that both indices are integers, we can return them as such
    batch_index = int(indices[0])
    sample_index = int(indices[1])
    return batch_index, sample_index


def get_index_map(filenames):
    indexes = {}

    index = 0

    # Iterate over all filenames
    for filename in filenames:
        # Extract the batch and sample indices
        batch_index, sample_index = extract_indices(filename)

        if (batch_index, sample_index) not in indexes.values():
            indexes[index] = (batch_index, sample_index)
            index += 1

    return indexes
