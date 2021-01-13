import pandas as pd
import numpy as np
import h5py
import os
from pathlib import Path

#Data folder with all the images
data_folder = '/media/HDD/pgonzalez/data'
#Here is the dataset information, basically samples, class
ifcb_csv = 'IFCB.csv'
#Here is where we will put the hdf5 files
output_folder = 'output'


def create_h5py_file(sample):
    '''
    This function gets a dataframe with the examples of only one sample

    We will create an hdf5 file with this images

    Each image will be a dataset in the file with extra attribues: AutoClass, OriginalClass
    '''

    samplename = sample['Sample'][0]
    print("Processing sample {}".format(samplename))
    f = h5py.File(os.path.join(output_folder,samplename+".hdf5"), "w")

    for _, example in sample.iterrows():
        with open(example['path'], 'rb') as img_f:
            image_file = img_f.read()
        img_np_array = np.asarray(image_file)
        dset = f.create_dataset(Path(example['path']).stem, data=img_np_array)
        #Add the attributes
        dset.attrs['AutoClass'] = example['AutoClass']
        dset.attrs['OriginalClass'] = example['OriginalClass']
        dset.attrs['FunctionalGroup'] = example['FunctionalGroup']

    f.close()


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#else:
#    raise ValueError("The output folder already exists, doing nothing")

#Load the information file
ifcb_csv = pd.read_csv('IFCB.csv',header=0,sep=',',quotechar='"')
print(ifcb_csv)

#compute file names
ifcb_csv['year'] = ifcb_csv['Sample'].str[6:10].astype(str) #Compute the year
ifcb_csv['path']=data_folder+'/'+ifcb_csv['year']+'/'+ifcb_csv['OriginalClass'].astype(str)+'/'+ifcb_csv['Sample'].astype(str)+'_'+ifcb_csv['roi_number'].apply(lambda x: str(x).zfill(5))+'.png'

#build the hdf5 files for each sample
ifcb_csv.groupby('Sample').apply(create_h5py_file)

