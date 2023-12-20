import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_test_dataloader(focal_height=1.0, image_resolution=(128, 128)):
    return _get_dataloader("./data/test", focal_height, image_resolution)


def get_train_dataloader(focal_height=1.0, image_resolution=(128, 128)):
    return _get_dataloader("./data/train", focal_height, image_resolution)


def _get_dataloader(main_dir, focal_height, image_resolution):
    transform = transforms.Compose([
        transforms.Resize(image_resolution),
        transforms.ToTensor(),
    ])

    data = AosDataset(main_dir, transform, focal_height)

    return DataLoader(data, batch_size=16, shuffle=True)


class AosDataset(Dataset):
    def __init__(self, main_dir, transform, focal_height):
        self.main_dir = main_dir
        self.transform = transform
        self.focal_height = focal_height
        all_imgs = glob.glob(os.path.join(main_dir, '*.png'))
        self.total_imgs = sorted([img for img in all_imgs if "aos_thermal" in img])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_name = self.total_imgs[idx]
        gt_name = img_name.replace("-aos_thermal", "_GT_pose_0_thermal")
        image = Image.open(img_name).convert("L")
        gt_image = Image.open(gt_name).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            gt_image = self.transform(gt_image)

        return image, gt_image
