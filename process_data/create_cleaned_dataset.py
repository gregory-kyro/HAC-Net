import pandas as pd
import numpy as np
import seaborn as sns
import os

def create_cleaned_dataset(PDBbind_dataset_path, general_set_PDBs_path, refined_set_PDBs_path, plot = False):
  """
  Produces a csv file containing PDB id, binding affinity, and set (general/refined)
  
  Inputs:
  1) PDBbind_dataset_path: path to PDBbind dataset; dataset is included in github repository as 'PDBbind_2020_data.csv'
  2) general_set_PDBs_path: path to PDBbind general set excluding refined PDBs
  3) refined_set_PDBs_path: path to PDBbind refined set PDBs
  4) plot = True will generate a plot of density as a function of binding affinity for general
     and refined sets
     
  Output:
  1) A cleaned csv containing PDB id, binding affinity, and set (general/refined):
     'PDBbind_2020_data_cleaned.csv'
  """
  
  # load dataset
  data = pd.read_csv(PDBbind_dataset_path)
  
  # check for NaNs in affinity data
  if data['-log(Kd/Ki)'].isnull().any() != False:
    print('There are NaNs present in affinity data!')
    
  # create list of PDB id's
  pdbid_list = list(data['PDB ID'])
  
  # remove affinity values that do not have structural data by searching PDBs
  missing = []
  for i in range(len(pdbid_list)):
    pdb = pdbid_list[i]
    if os.path.isdir(str(general_set_PDBs_path) + pdb)==False and os.path.isdir(str(refined_set_PDBs_path) + pdb)==False:
        missing.append(pdb)
  data = data[~np.in1d(data['PDB ID'], list(missing))]

  # distinguish PDB id's in general and refined sets
  general_dict = {}
  refined_dict = {}
  for i in range(len(pdbid_list)):
    pdb = pdbid_list[i]
    if os.path.isdir(str(general_set_PDBs_path) + pdb)==True:
        general_dict[pdb] = 'general'
    if os.path.isdir(str(refined_set_PDBs_path) + pdb)==True:
        refined_dict[pdb] = 'refined'
   
  # add 'set' column to data and fill with 'general' or 'refined'
  data['set'] = np.nan
  data.loc[np.in1d(data['PDB ID'], list(general_dict)), 'set'] = 'general'
  data.loc[np.in1d(data['PDB ID'], list(refined_dict)), 'set'] = 'refined'
  
  # write out csv of cleaned dataset
  data[['PDB ID', '-log(Kd/Ki)', 'set']].to_csv('PDBbind_2020_data_cleaned.csv', index=False)
  
  # read in and view the cleaned dataset
  display(pd.read_csv('PDBbind_2020_data_cleaned.csv'))
  
  if plot == True:
    # plot affinity distributions for general and refined sets
    grid = sns.FacetGrid(data, row='set', row_order=['general', 'refined'],
                     size=3, aspect=2)
    grid.map(sns.distplot, '-log(Kd/Ki)')
  else:
    return
