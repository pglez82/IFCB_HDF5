import pandas as pd

metadata = pd.read_csv('metadata_raw.csv',index_col=0)
print(metadata)
metadata.loc[metadata.index[0:21], 'scatteringPhotomultiplierSetting'] = 0.4
metadata.loc[metadata.index[0:21], 'fluorescencePhotomultiplierSetting'] = 0.7
metadata.loc[metadata.index[0:21], 'blobSizeThreshold'] = 10
metadata.loc[metadata.index[0:10], 'temperature'] = metadata.loc[metadata.index[10], 'temperature']
metadata.loc[metadata.index[0:10], 'humidity'] = metadata.loc[metadata.index[10], 'humidity']
metadata.loc[metadata.index[0:10], 'binarizeThreshold'] = metadata.loc[metadata.index[10], 'binarizeThreshold']
print(metadata)
metadata.to_csv('metadata.csv')
