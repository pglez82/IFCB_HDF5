import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# https://ifcb-data.whoi.edu/api/metadata/IFCB5_2011_081_200557

def download_metadata(row):
    url = 'https://ifcb-data.whoi.edu/api/metadata/'+row.name
    response = requests.get(url)
    data = response.json()['metadata']
    if 'temperature' in data:
        row['temperature'] = data['temperature'] 
    if 'humidity' in data:
        row['humidity'] = data['humidity']
    if 'binarizeThreshold' in data:
        row['binarizeThreshold'] = data['binarizeThreshold']
    if 'scatteringPhotomultiplierSetting' in data:
        row['scatteringPhotomultiplierSetting'] = data['scatteringPhotomultiplierSetting']
    if 'fluorescencePhotomultiplierSetting' in data:        
        row['fluorescencePhotomultiplierSetting'] = data['fluorescencePhotomultiplierSetting']
    if 'blobSizeThreshold' in data:
        row['blobSizeThreshold'] = data['blobSizeThreshold']
    return row



ifcb_csv = pd.read_csv('IFCB.csv',header=0,sep=',',quotechar='"')
samples = ifcb_csv['Sample'].unique()

metadata = pd.DataFrame(columns=('year','day','hour','minute','second', 'temperature','humidity','binarizeThreshold','scatteringPhotomultiplierSetting','fluorescencePhotomultiplierSetting','blobSizeThreshold','device'), index=samples)

#compute file names
metadata['year'] = metadata.index.str[6:10].astype(int) #Compute the year
metadata['day'] = metadata.index.str[11:14].astype(int) #Compute the day
metadata['hour'] = metadata.index.str[15:17].astype(int) #Compute the hour
metadata['minute'] = metadata.index.str[17:19].astype(int) #Compute the minute
metadata['second'] = metadata.index.str[19:21].astype(int) #Compute the second
metadata['device'] = np.where(metadata.index.str[0:5].astype(str) == 'IFCB5', 5, 1) #Compute the device

print(metadata)

#build the hdf5 files for each sample
tqdm.pandas()
metadata = metadata.progress_apply(download_metadata, axis=1)
metadata.to_csv('metadata_raw.csv')

