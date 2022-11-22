''' This process is largely inspired by Pafnucy <https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/prepare.py>, 
    utilizing the tfbio package <https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py>''' 

def create_hdf(affinity_data_path, output_total_hdf, mol2_path, general_PDBs_path, refined_PDBs_path, output_train_hdf, output_val_hdf, output_test_hdf, path_to_elements_xml,  val_fraction, test_fraction, core_test, bad_pdbids_input = []):
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
    10) approximate fraction of the data to be reserved for validation
    11) approximate fraction of the data to be reserved for testing
    12) boolean flag indicating whether or not the 2016 PDBBind core set should be used for the testing set.
    13) bad_pdbids_input, an array containing any pdbids that crashed chimera or crashed this function. Set to [] by default
 
    output:
    1)  a complete hdf file containing featurized data for all of the PDB id's that will be used, saved as:
        'path/to/output/hdf/file.hdf'
    2)  three hdf files containing the featurized data for the PDB id's that will be used in training, validation, and testing sets, saved as:
        'path/to/output/training/hdf/file.hdf', 'path/to/output/validation/hdf/file.hdf', and 'path/to/output/testing/hdf/file.hdf', respectively
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
    #os.chdir(path/to/tfbio)     change to tfbio directory
    from tfbio.data import Featurizer

    # define function to select pocket mol2 files with atoms that have unrealistic charges
    def high_charge(molecule):
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:
                if (abs(atom.__getattribute__('partialcharge'))>2):
                    return True
                else: 
                    return False  

    # define function to extract features from the binding pocket mol2 file check for unrealistic charges
    def __get_pocket():
        for pfile in pocket_files:
            pocket = next(openbabel.pybel.readfile('mol2', pfile))
            if high_charge(pocket):
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
            if high_charge(ligand):
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

    #Create a list of all complexes in the 2016 PDBbind core set
    core_set=['3ao4', '3gv9', '1uto', '1ps3', '4ddk', '4jsz', '3g2z', '3dxg', '3l7b', '3gr2', '3kgp', '3fcq', '3lka', '3zt2', '3udh', '3g31', '4llx', '4u4s', '4owm', '5aba', '2xdl', '4kz6', '2ymd', '3aru', '1bcu', '3zsx', '4ddh', '4mrw', '4eky', '4mrz', '4abg', '5a7b', '3dx1', '4bkt', '2v00', '4cig', '3n7a', '3d6q', '2hb1', '3twp', '4agn', '1c5z', '3nq9', '4msn', '2w66', '3kwa', '3g2n', '4cr9', '4ih5', '4de2', '3ozt', '3f3a', '1a30', '3ivg', '3u9q', '3rsx', '3pxf', '2wbg', '3rr4', '4w9c', '3mss', '4agp', '4mgd', '1vso', '4jxs', '1q8t', '3acw', '4lzs', '3r88', '4ciw', '2w4x', '2brb', '1p1q', '3d4z', '1bzc', '1nc3', '4agq', '4w9l', '2yge', '5c1w', '2r9w', '3gy4', '3syr', '3zso', '2br1', '1s38', '3b27', '4gkm', '4m0z', '1w4o', '3ueu', '4ih7', '4jfs', '3ozs', '3bv9', '1gpk', '1syi', '2cbv', '1ydr', '4de3', '3coz', '2wca', '3u5j', '4dli', '1z9g', '3arv', '3n86', '5c28', '4j28', '3jvr', '1o5b', '2y5h', '3qqs', '3wz8', '4dld', '3ehy', '3uev', '3ebp', '1o0h', '1q8u', '4de1', '4msc', '4w9i', '3ary', '3coy', '3f3c', '2fxs', '4kzq', '2qnq', '1nc1', '2wvt', '1yc1', '3bgz', '4wiv', '3k5v', '4eor', '3uew', '2wnc', '2zb1', '2qbr', '3arq', '2j78', '4ea2', '1r5y', '4m0y', '1gpn', '2weg', '4kzu', '4mme', '3cj4', '3uo4', '3wtj', '3jvs', '1k1i', '2yfe', '4k77', '2xj7', '2iwx', '4f09', '4djv', '4w9h', '4ogj', '1p1n', '3dx2', '2xnb', '3n76', '3pyy', '2zcr', '3oe5', '3jya', '3gbb', '3uex', '4f9w', '2wer', '1lpg', '3zdg', '1z95', '1pxn', '3arp', '3f3d', '3tsk', '2j7h', '2xii', '4cra', '4gfm', '1oyt', '3p5o', '3gc5', '2vvn', '1qf1', '1ydt', '3pww', '1owh', '2zy1', '3up2', '4j21', '2xys', '2qbq', '3oe4', '3rlr', '2xb8', '2c3i', '4e5w', '3f3e', '1u1b', '3qgy', '3ryj', '4j3l', '3prs', '4pcs', '4hge', '1o3f', '2qe4', '3uuo', '3cyx', '3e92', '3fur', '2cet', '5tmn', '3ag9', '3kr8', '3nx7', '3fv2', '4eo8', '3e5a', '1nvq', '2v7a', '4x6p', '1h23', '4e6q', '2al5', '2qbp', '2zda', '3b68', '2xbv', '3b1m', '2fvd', '2vw5', '2wn9', '3ejr', '4qd6', '3u8k', '3ge7', '4crc', '4ivb', '2vkm', '2wtv', '3b5r', '2zcq', '3e93', '4k18', '2p4y', '3dd0', '3nw9', '3ui7', '3uri', '1qkt', '1h22', '3gnw', '1sqa', '4jia', '3b65', '3fv1', '4qac', '2yki', '3g0w', '4ivd', '4ty7', '2pog', '4gr0', '1eby', '1z6e', '1e66', '4ivc', '4twp', '4rfm', '1y6r', '3u8n', '4tmn', '2p15', '3myg', '4gid', '3utu', '5c2h', '1mq6', '5dwr', '4f2w', '2x00', '3o9i', '4f3c']

    # read cleaned affinity data into a pandas DataFrame
    affinity_data_cleaned=affinities[~affinities['pdbid'].isin(bad_complexes)]

    affinity_data_cleaned_minus_core=affinity_data_cleaned[~affinity_data_cleaned['pdbid'].isin(core_set)]

    # create a column for the percentile of affinities
    affinity_data_cleaned['Percentile']= pd.qcut(affinity_data_cleaned['-logKd/Ki'].rank(method='first'),
                                q = 100, labels = False)
    
    affinity_data_cleaned_minus_core['Percentile']= pd.qcut(affinity_data_cleaned_minus_core['-logKd/Ki'].rank(method='first'),
                                q = 100, labels = False)
                                
    # compute number of complexes to sample from each percentile for validation and testing, respectively
    val_to_sample = round(val_fraction*num_pockets/100)
    test_to_sample= round(test_fraction*num_pockets/100)

    # create empty arrays
    test_pdbids = []
    val_pdbids = []
    
    #assemble lists of pdbids for validation and test sets
    for i in range(0, 100):
        if core_test:
            temp = affinity_data_cleaned_minus_core[affinity_data_cleaned_minus_core['Percentile'] == i]
        else:
            temp = affinity_data_cleaned[affinity_data_cleaned['Percentile'] == i]
        num_vals = len(temp)
        val_rows = temp.sample(val_to_sample)
        val_pdbids = np.hstack((val_pdbids, (val_rows['pdbid']).to_numpy()))
        if core_test:
            continue
        else:
            for pdbid in (val_rows['pdbid']).to_numpy():
                temp = temp[temp.pdbid != pdbid]
            test_rows = temp.sample(test_to_sample)
            test_pdbids = np.hstack((test_pdbids, (test_rows['pdbid']).to_numpy()))

    if core_test:
        test_pdbids=core_set

    # populate the test and validation hdf files by transferring those datasets from the total file
    with h5py.File(output_test_hdf, 'w') as g, h5py.File(output_val_hdf, 'w') as h:
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
    with h5py.File(output_train_hdf, 'w') as g:
        with h5py.File(output_total_hdf, 'r') as f:
            for pdbid in affinity_data_cleaned['pdbid'].to_numpy():
                if pdbid not in holdouts:
                    ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
                    ds.attrs['affinity'] = f[pdbid].attrs['affinity']
                    ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
