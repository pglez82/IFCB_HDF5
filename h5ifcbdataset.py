from torch.utils.data import Dataset
import h5py
import numpy as np
import io,sys
from PIL import Image
from pathlib import Path


class H5IFCBDataset(Dataset):
    def __init__(self, files, classes,classattribute,transform=None):
        self.transform = transform
        self.classes = classes
        self.classes.sort()
        self.files = files
        self.files.sort() #Load them in order so we get always the same results
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.targets = []
        self.images = []
        self.samples = []

        #Open the files and store the images into memory
        for file in files:
            f = h5py.File(file, 'r')
            for example in f.keys():
                cl = f[example].attrs[classattribute]
                self.targets.append(self.class_to_idx[cl])
                self.images.append(np.array(f[example]))
                self.samples.append(Path(file).stem)

            f.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(io.BytesIO(self.images[index]))
        img = img.convert('RGB')
        target = int(self.targets[index])
        sample = self.samples[index]
        if self.transform is not None:
            img = self.transform(img)
        return img,target,sample
