import os
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from util.mean_std import mean, std


def _to_tensor(img):
    # Plain reimplementation of torchvision.transforms.functional.to_tensor (PIL RGB image
    # -> [C, H, W] float tensor in [0, 1]) - avoids depending on torchvision.transforms here.
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _normalize(tensor, mean, std):
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


class MyDataset_demo(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_files = os.listdir(img_path)

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file_name = self.img_files[index]
        img = Image.open(os.path.join(self.img_path, img_file_name)).convert('RGB')

        img = self.scale_image_to_512(img)

        canvas = Image.new('RGB', (512, 512), (255, 255, 255))

        img_width, img_height = img.size
        upper_left_x = (512 - img_width) // 2
        upper_left_y = (512 - img_height) // 2

        canvas.paste(img, (upper_left_x, upper_left_y))

        img_tensor = _to_tensor(canvas)
        img_tensor = _normalize(img_tensor, mean=mean, std=std)



        target = {'edges': None, 'image_id': None, 'size': None, 'semantic_left_up': None, 'semantic_right_up': None, 'semantic_right_down': None, 'semantic_left_down': None, 'unnormalized_points': None, 'points': None, 'layer_indices': None, 'graph': None}

        return img_tensor, target

    def scale_image_to_512(self, img):
        scale_factor = 512.0 / max(img.size)
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        return img
