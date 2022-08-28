**Procedure for Processing and Preparing Data**

**1)** Read in PDBbind dataset and create a csv file to be used to produce hdf5 files
  - parse_PDBbind_data.py
  - run this code to perform step 1:
        
        from parse_PDBbind_data import create_cleaned_dataset
        
        create_cleaned_dataset('path/to/dataset.csv', 'path/to/general/set/pdbs/%s', 'path/to/refined/set/pdbs/%s', plot=True)
