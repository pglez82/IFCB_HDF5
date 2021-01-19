from torch.utils.data import Dataset
import h5py
import numpy as np
import io,sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm


class H5IFCBDataset(Dataset):
    def __init__(self, files, classes,classattribute,verbose=0,defaultclass="mix",transform=None):
        '''
        Inits the dataset
        :param list files: List of HDF5 files to load
        :param list classes: Sorted list with the classes of the problem as strings (must be coherent with the next attribute)
        :param str classattribute: Attribute in the hdf5 files that contains the class. For IFCB can be: AutoClass, OriginalClass or FunctionalGroup
        :param int verbose: Show information of the process
        :param transform: transformations. See test.py for an example. At least the transformation to Tensor should be there
        :return: 
        '''
        self.verbose = verbose
        self.transform = transform
        if (type(classes) is np.ndarray and (classes != np.sort(classes)).any()):
            raise ValueError("classes attribute should be a numpy array sorted")
        self.classes = classes
        self.classes.sort()
        self.files = files
        self.files.sort() #Load them in order so we get always the same results
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.targets = []
        self.images = []
        self.samples = []

        #Open the files and store the images into memory
        for i in tqdm(range(len(files)),desc='Loading samples: ',disable=(self.verbose<1)):
            file = files[i]
            f = h5py.File(file, 'r')
            for example in f.keys():
                cl = f[example].attrs[classattribute]
                #What if the examples have a class that is not in the list of classes given
                if cl not in self.class_to_idx:
                    self.targets.append(self.class_to_idx[defaultclass])
                else:
                    self.targets.append(self.class_to_idx[cl])
                self.images.append(np.array(f[example]))
                self.samples.append(Path(file).stem)

            f.close()
        #Check that we have examples of all classes
        real_classes=np.unique(self.targets)
        real_classes.sort()
        missing_classes = np.setdiff1d(np.arange(len(classes)),real_classes)
        if missing_classes.size>0:
                print("ERROR. YOU MUST REBUILD THE DATASET! Not all the classes are in the training set: {}. Remove them from the classes list and start again.".format(missing_classes))
    
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
