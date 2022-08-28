## Procedure for Processing and Preparing Data

### 1) Read in PDBbind dataset and create a csv file to be used to produce hdf5 files
  - **parse_PDBbind_data.py**
        
### 2) Add hydrogens to pocket PDB files and convert to mol2 file type using Chimera 1.16, remove TIP3P atoms from mol2 files
  - **add_H_and_mol2_chimera.py**
  - **remove_tip3p_chimera.sh**
  
### 3) Compute charges for mol2 files externally with Atomic Charge Calculator II (ACC2)
  - **ACC2: <https://acc2.ncbr.muni.cz/>**
   -------------
**4)** Identify non-problematic complexes, and create train, test, and validation hdf files
  - convert_to_hdf.py
  
**5)** Create hdf files containing voxelized train, test, and validation data for use in the 3dcnn
  - create_voxelized_hdf.py
