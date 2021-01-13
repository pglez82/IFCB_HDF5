# IFCB_HDF5
IFCB dataset in the HDF5 format

## Objetives

This project will consist on two parts:
- Python script to convert all the IFCB data to the HDF5 format (`maker.py`). The idea is to create one hdf5 file per sample (i.e., group of examples captured in the same time frame).
- Dataset class for using the dataset in pytorch (`h5ifcbdataset.py`). Loads the samples indicated in memory. The idea here is to load the byte arrays (compressed images) and then when the DataLoader asks for some elements, it gets the byte array from memory and converts it to an image (an then to a tensor, using the given transformations)
