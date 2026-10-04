import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .functions import sort_key


class RedEdgeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.band_names = ["blue", "green", "red", "near_ir", "red_edge"]
        self.data_dir = data_dir
        self.transform = transform

        # Get all the directories that directly contain data
        self.content_dirs = [os.path.join(data_dir, content_dir) 
                             for content_dir in os.listdir(data_dir)]

        # Merge images from different folders to one list
        self.contents = []
        for content_dir in self.content_dirs:
            current_contents = [os.path.join(content_dir, current_content) 
                                for current_content in os.listdir(content_dir)]
            self.contents += current_contents

        # Sort and reshape to pack up 5 images taken that the same position
        self.contents.sort(key=sort_key)
        self.contents = np.reshape(self.contents, (-1, 5))

    def __len__(self):
        return len(self.contents)

    # Return 5 images at a time
    def __getitem__(self, idx):
        images = dict()
        for ite, content in enumerate(self.contents[idx]):
            image = Image.open(content)
            image = np.array(image) / 2**16
            images[self.band_names[ite]] = self.transform(image)
            
        return images
