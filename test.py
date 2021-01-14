import h5ifcbdataset
import pandas as pd
import torchvision.transforms as T

ifcb_csv = pd.read_csv('IFCB.csv',header=0,sep=',',quotechar='"')
classes = pd.unique(ifcb_csv['AutoClass'])
classes.sort()

#Get all the samples of 2006
years = ['2006']
ifcb_csv['year'] = ifcb_csv['Sample'].str[6:10].astype(str) #Compute the year
samples=ifcb_csv.groupby('Sample').first()
print(len(samples))
files = list(samples[samples['year'].isin(years)].index) 
files = ['output/'+f+'.hdf5' for f in files]

train_transform = T.Compose([
  T.Resize(size=256),
  T.RandomResizedCrop(size=224),
  T.RandomHorizontalFlip(),
  T.ToTensor(),            
  #T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

dataset = h5ifcbdataset.H5IFCBDataset(files,classes,classattribute="AutoClass",verbose=1,transform=train_transform)
print(len(dataset))
image,label,sample = dataset[1000]
print(type(image))
print(label)
print(sample)
