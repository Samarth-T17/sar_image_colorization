from torch.utils.data import Dataset
from transforms import both_transform, transform_only_input, transform_only_mask
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import torch


def local_mean_variance(image, window_size=3):
    kernel = torch.ones((1, 1, window_size, window_size), dtype=torch.float32) / (window_size ** 2)
    image = image.unsqueeze(0).unsqueeze(0).float()
    local_mean = F.conv2d(image, kernel, padding=window_size // 2)
    squared_diff = (image - local_mean) ** 2
    local_variance = F.conv2d(squared_diff, kernel, padding=window_size // 2)
    return local_mean.squeeze(), local_variance.squeeze()
    

def overall_variance_with_enl(image, enl):
    overall_mean = torch.mean(image)
    overall_var = (overall_mean ** 2) / enl
    return overall_var
    

def lee_filter(image,enl,window_size):
    image = image.float()
    local_mean, local_variance = local_mean_variance(image, window_size)
    overall_var = overall_variance_with_enl(image, enl)
    weight = local_variance / (local_variance + overall_var)
    filtered_image = local_mean + weight * (image - local_mean)
    return filtered_image.numpy()
    

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_dirs = {
            "grassland": os.path.join(root_dir, "grassland", "s1"),
            "urban": os.path.join(root_dir, "urban", "s1"),
            "barrenland": os.path.join(root_dir, "barrenland", "s1"),
            "agri": os.path.join(root_dir, "agri", "s1"),
        }
        self.target_dirs = {
            "grassland": os.path.join(root_dir, "grassland", "s2"),
            "urban": os.path.join(root_dir, "urban", "s2"),
            "barrenland": os.path.join(root_dir, "barrenland", "s2"),
            "agri": os.path.join(root_dir, "agri", "s2"),
        }
        self.list_files = []
        for category in self.input_dirs:
            input_files = os.listdir(self.input_dirs[category])
            for file_name in input_files:
                s2 = file_name[:15] + '2' + file_name[15 + 1:]
                target_img_path = os.path.join(self.target_dirs[category], s2)
                if os.path.exists(target_img_path):
                    self.list_files.append((category, file_name))

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        category, img_file = self.list_files[index]
        en = -1.0
        
        if category == 'agri':
            en = 0.0
        elif category == 'barrenland':
            en = 1.0
        elif category =='urban':
            en = 2.0
        else: 
              en=3.0
        
        s2 = img_file[:15] + '2' + img_file[15 + 1:]
        input_img_path = os.path.join(self.input_dirs[category], img_file)
        target_img_path = os.path.join(self.target_dirs[category], s2)
        input_image = np.array(Image.open(input_img_path))
        target_image = np.array(Image.open(target_img_path))
        filtered_image=lee_filter(torch.from_numpy(input_image),140,3)
        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        input_image = transform_only_input(image=input_image)["image"]
        filtered_image = transform_only_input(image=filtered_image)["image"]
        hot = torch.full((1, 256, 256), en)
        input_image = torch.cat([input_image, filtered_image, hot], dim = 0)
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image
