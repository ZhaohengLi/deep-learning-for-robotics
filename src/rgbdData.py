import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class RGBD(Dataset):
    def __init__(self, stage='train', transforms=None):
        self.root_dir = '../rgbd'
        self.transforms = transforms
        self.stage = stage
        self.train_val_ratio = 0.6
        self.samples = []
        for i_label, label in enumerate(sorted(os.listdir(self.root_dir))):
            lbpath = os.path.join(self.root_dir, label)
            all_instances = sorted(os.listdir(lbpath))
            train_instances = all_instances[:int(self.train_val_ratio*len(all_instances))]
            val_instances = all_instances[int(self.train_val_ratio*len(all_instances)):]
            if self.stage == 'train':
                instances = train_instances
            else:
                instances = val_instances
            for instance in instances:
                instance_path = os.path.join(self.root_dir, label, instance)
                imgs = sorted(os.listdir(instance_path))
                for img in imgs:
                    name_info = img.split('_')
                    if name_info[-1] == 'crop.png':
                        img_path = os.path.join(instance_path, img)
                        dep_path = os.path.join(instance_path, img.replace('crop.png', 'depthcrop.png'))
                        self.samples.append([img_path, dep_path, i_label])
                         
             
    def __getitem__(self, index):
        sample = self.samples[index]
        img = Image.open(sample[0])
        dep = Image.open(sample[1])
        if self.transforms is not None:
            img = self.transforms(img)
            dep = self.transforms(dep)
        target = sample[2]
        return img, dep, target
         
    def __len__(self):
        return len(self.samples)
