import h5ifcbdataset
import pandas as pd
import torchvision.transforms as T

ifcb_csv = pd.read_csv('IFCB.csv',header=0,sep=',',quotechar='"')
classes = pd.unique(ifcb_csv['AutoClass'])
classes.sort()

train_transform = T.Compose([
  T.Resize(size=256),
  T.RandomResizedCrop(size=224),
  T.RandomHorizontalFlip(),
  T.ToTensor(),            
  #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

dataset = h5ifcbdataset.H5IFCBDataset(['output/IFCB1_2006_158_000036.hdf5'],classes,classattribute="AutoClass",transform=train_transform)
print(len(dataset))
image,label,sample = dataset[1000]
print(type(image))
print(label)
print(sample)
