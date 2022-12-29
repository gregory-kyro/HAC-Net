''' This process is largely inspired by Pafnucy <https://gitlab.com/cheminfIBB/pafnucy/-/blob/master/prepare.py>, 
    utilizing the tfbio package <https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py>''' 

def create_hdf(affinity_data_path, output_total_hdf, mol2_path, general_PDBs_path, refined_PDBs_path, path_to_elements_xml,  bad_pdbids_input = []):
    """
    This function converts the mol2 files into one hdf file containing all complexes provided.
    input:
    1) path/to/cleaned/affinity/data.csv
    2) path/to/total/output/hdf/file.hdf
    3) path/to/mol2/files
    4) path/to/PDBs/in/general_set
    5) path/to/PDBs/in/refined_set
    6) path/to/elements.xml
    7) bad_pdbids_input, an array containing any pdbids that crashed chimera or crashed this function. Set to [] by default
 
    output:
    1)  a complete hdf file containing featurized data for all of the PDB id's that will be used, saved as:
        'path/to/output/hdf/file.hdf'
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
