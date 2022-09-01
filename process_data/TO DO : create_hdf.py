''' This process is largely inspired by Pafnucy <url to pafuncy>, utilizing the tfbio package <url to tfbio package>''' ### this line subject to change
'''Inspired by the following code: https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/prepare.py'''

def convert_to_hdf(affinity_data_path, output_total_hdf, mol2_path, general_PDBs_path, refined_PDBs_path, output_train_hdf, output_val_hdf, output_test_hdf, path_to_elements_xml,  val_fraction, test_fraction, num_splits=1, bad_pdbids_input = []):
    """
    This function converts the mol2 files into three cleaned hdf files containing datasets for training, testing, and validation complexes, respectively.
    input:
    1) path/to/cleaned/affinity/data.csv
    2) path/to/total/output/hdf/file.hdf
    3) path/to/mol2/files
    4) path/to/PDBs/in/general_set
    5) path/to/PDBs/in/refined_set
    6)  path/to/output/training/hdf/file.hdf
    7) path/to/output/validation/hdf/file.hdf
    8) path/to/output/testing/hdf/file.hdf
    9) path/to/elements.xml
    10) fraction of the data to be reserved for validation
    11) fraction of the data to be reserved for testing
    12) number of splits to create (with non-overlapping test sets)
    13) bad_pdbids_input, an array containing any pdbids that crashed chimera or crashed this function (roughly 1 in 3,000 files). Set to [] by default
 
    output:
    1)  a complete hdf file containing featurized data for all of the PDB id's that will be used, saved as:
        'path/to/output/hdf/file.hdf'
    2)  three hdf files containing the featurized data for the PDB id's that will be used in validation, training, and testing sets, saved as:
        'path/to/output/validation/hdf/file.hdf', 'path/to/output/training/hdf/file.hdf', and 'path/to/output/testing/hdf/file.hdf', respectively
    """

    #Necessary import statements
    import pickle
    import numpy as np
    import openbabel.pybel
    from openbabel.pybel import Smarts
    from math import ceil, sin, cos, sqrt, pi
    from itertools import combinations
    import pandas as pd
    import h5py
    import csv
    import xml.etree.ElementTree as ET
    import os

    # Import the Featurizer class from tfbio(source code here): https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py
    #from tfbio.data import Featurizer

    
    # define function to select pocket mol2 files with atoms that have unrealistic charges
    def get_charge(molecule):
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:
                if (abs(atom.__getattribute__('partialcharge'))>2):
                    return 'bad_complex'
                else: 
                    return 'no_error'  

    # define function to extract features from the binding pocket mol2 file check for unrealistic charges
    def __get_pocket():
        for pfile in pocket_files:
            pocket = next(openbabel.pybel.readfile('mol2', pfile))
            if(get_charge(pocket)==('bad_complex')):
                bad_complexes.append((os.path.splitext(os.path.split(pfile)[1])[0]).split('_')[0]) 
            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            pocket_vdw = parse_mol_vdw(mol=pocket, element_dict=element_dict)
            yield (pocket_coords, pocket_features, pocket_vdw)

    # define function to extract information from elements.xml file
    def parse_element_description(desc_file):
        element_info_dict = {}
        element_info_xml = ET.parse(desc_file)
        for element in element_info_xml.getiterator():
            if "comment" in element.attrib.keys():
                continue
            else:
                element_info_dict[int(element.attrib["number"])] = element.attrib

        return element_info_dict

    # define function to create a list of van der Waals radii for a molecule
    def parse_mol_vdw(mol, element_dict):
        vdw_list = []
        for atom in mol.atoms:
            if int(atom.atomicnum)>=2:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))
        return np.asarray(vdw_list)



    # read in data and format properly
    element_dict = parse_element_description(path_to_elements_xml)
    affinities = pd.read_csv(affinity_data_path)
    pdbids_cleaned = affinities['pdbid'].to_numpy()
    bad_complexes = bad_pdbids_input
    

    # fill lists with paths to pocket and ligand mol2 files
    pocket_files, ligand_files = [], []
    for i in range(0, len(pdbids_cleaned)):
        if pdbids_cleaned[i] not in bad_complexes:
            pocket_files.append(mol2_path + "/" + pdbids_cleaned[i] + '_pocket.mol2')
            if affinities['set'][i]=='general':
                ligand_files.append(general_PDBs_path + "/" + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')
            else:
                ligand_files.append(refined_PDBs_path + "/" + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')

    num_pockets = len(pocket_files)
    num_ligands = len(ligand_files)


    affinities_ind = affinities.set_index('pdbid')['-logKd/Ki']

    featurizer = Featurizer()


    # create a new hdf file to store all of the data
    with h5py.File(output_total_hdf, 'a') as f:

        pocket_generator = __get_pocket()
        for lfile in ligand_files:
            # use pdbid as dataset name
            name = os.path.splitext(os.path.split(lfile)[1])[0]
            pdbid = name.split('_')[0]

            #Avoid duplicates
            if pdbid in list(f.keys()):
                continue

            # read ligand file using pybel
            ligand = next(openbabel.pybel.readfile('mol2', lfile))


            # extract features from pocket and check for unrealistic charges
            pocket_coords, pocket_features, pocket_vdw = next(pocket_generator)

            # extract features from ligand and check for unrealistic charges
            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)
            if(get_charge(ligand)=='bad_complex'):
                if pdbid not in bad_complexes:
                    bad_complexes.append(pdbid)

            # if the current ligand file is part of a bad complex, do not copy to the cleaned hdf file
            if pdbid in bad_complexes:
                continue

            # center the ligand and pocket coordinates
            centroid = ligand_coords.mean(axis=0)
            ligand_coords -= centroid
            pocket_coords -= centroid

            # assemble the features into one large numpy array: rows are heavy atoms, columns are coordinates and features
            data = np.concatenate(
                (np.concatenate((ligand_coords, pocket_coords)),
                np.concatenate((ligand_features, pocket_features))),
                axis=1,
            )
            # concatenate van der Waals radii into one numpy array
            vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))


            # create a new dataset for this complex in the hdf file
            dataset = f.create_dataset(pdbid, data=data, shape=data.shape,
                                        dtype='float32', compression='lzf')

            # add the affinity and van der Waals radii as attributes for this dataset 
            dataset.attrs['affinity'] = affinities_ind.loc[pdbid]
            assert len(vdw_radii) == data.shape[0]
            dataset.attrs["van_der_waals"] = vdw_radii



    # read cleaned affinity data into a pandas DataFrame
    affinity_data_cleaned=affinities[~affinities['pdbid'].isin(bad_complexes)]

    # create a column for the percentile of affinities
    affinity_data_cleaned['Percentile']= pd.qcut(affinity_data_cleaned['-logKd/Ki'],
                                q = 100, labels = False)
                                
    

    # compute number of complexes to sample from each percentile for validation and testing, respectively
    val_to_sample = round(val_fraction*num_pockets/100)
    test_to_sample= round(test_fraction*num_pockets/100)


    #Divide up the total number of complexes into the given number of splits
    previous_test_pdbids=[]

    for rep in range (0, num_splits):
        # create empty arrays
        test_pdbids = []
        val_pdbids = []
    
        #generate names for output hdf files
        output_train_hdf_temp = output_train_hdf[:-4] + str(rep) + '.hdf'
        output_val_hdf_temp= output_val_hdf[:-4] + str(rep) + '.hdf'
        output_test_hdf_temp= output_test_hdf[:-4] + str(rep) + '.hdf'

        #assemble lists of pdbids for validation and test sets
        for i in range(0, 100):
            temp = affinity_data_cleaned[affinity_data_cleaned['Percentile'] == i]
            num_vals = len(temp)
            val_rows = temp.sample(val_to_sample)
            val_pdbids = np.hstack((val_pdbids, (val_rows['pdbid']).to_numpy()))
            for pdbid in (val_rows['pdbid']).to_numpy():
                temp = temp[temp.pdbid != pdbid]
            temp = temp[~temp.pdbid.isin(previous_test_pdbids)]
            test_rows = temp.sample(test_to_sample)
            test_pdbids = np.hstack((test_pdbids, (test_rows['pdbid']).to_numpy()))

        previous_test_pdbids.append(test_pdbids)

        # populate the test and validation hdf files by transferring those datasets from the total file
        with h5py.File(output_test_hdf_temp, 'w') as g, h5py.File(output_val_hdf_temp, 'w') as h:
            with h5py.File(output_total_hdf, 'r') as f:
                for pdbid in val_pdbids:
                    ds = h.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
                    ds.attrs['affinity'] = f[pdbid].attrs['affinity']
                    ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
                for pdbid in test_pdbids:
                    ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
                    ds.attrs['affinity'] = f[pdbid].attrs['affinity']
                    ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
                
        # populate the train hdf file by transferring all other datasets from the total file
        holdouts = np.hstack((val_pdbids,test_pdbids))
        with h5py.File(output_train_hdf_temp, 'w') as g:
            with h5py.File(output_total_hdf, 'r') as f:
                for pdbid in affinity_data_cleaned['pdbid'].to_numpy():
                    if pdbid not in holdouts:
                        ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
                        ds.attrs['affinity'] = f[pdbid].attrs['affinity']
                        ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
