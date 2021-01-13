# IFCB_HDF5
IFCB dataset in the HDF5 format

## Objetives

This project will consist on two parts:
- Python script to convert all the IFCB data to the HDF5 format. The idea is to create one hdf5 file per sample (aka, group of examples captured in the same time frame).
- Dataset class for using the dataset in pytorch. As the dataset is not to big it would be good to add the capability of loading the dataset in memory.
