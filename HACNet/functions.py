# import packages
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import pickle
import numpy as np
import openbabel.pybel
from openbabel.pybel import Smarts
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
import pandas as pd
import csv
import os
import random
import xml.etree.ElementTree as ET
import h5py
from sklearn.metrics import pairwise_distances
import torch
from torch.utils.data import Dataset
from torch_geometric.nn.conv import MessagePassing, GatedGraphConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.aggr import AttentionalAggregation
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from torch.nn.parallel import DataParallel
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import DataListLoader, Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset, DataLoader
from torch._C import NoneType
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import *
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, pairwise_distances
from pymol import cmd
import requests
from pymol import cmd
from IPython.display import Image, display

from .CNN import *
from .GCN import *
from .MLP import *

""" Define a function to test HACNet, the 3D-CNN, or one of the GCN components on an HDF file """
def predict_hdf(architecture, cnn_test_path, gcn0_test_path, gcn1_test_path, cnn_checkpoint_path, gcn0_checkpoint_path, gcn1_checkpoint_path):

    """
    Inputs:
    1) architecture: either "HACNet", "cnn", "gcn0", or "gcn1"    
    2) cnn_test_path: path to cnn test set npy file
    3) gcn0_test_path: path to gcn0 test set hdf file
    4) gcn1_test_path: path to gcn1 test set hdf file
    5) cnn_checkpoint_path: path to cnn checkpoint file
    6) gcn0_checkpoint_path: path to gcn0 checkpoint file
    7) gcn1_checkpoint_path: path to gcn1 checkpoint file
    Output:
    1) Print statement of summary of evaluation: RMSE, MAE, r^2, Pearson r, Spearman p, Mean, SD
    2) Correlation scatter plot of true vs predicted values
    """

    # set CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    device_name = "cuda:0"
    if use_cuda:
        device = torch.device(device_name)
        torch.cuda.set_device(int(device_name.split(':')[1]))
    else:   
        device = torch.device('cpu')  

    if architecture == 'HACNet' or 'cnn':
        # 3D-CNN
        batch_size = 50
        # load dataset
        cnn_dataset = MLP_Dataset(cnn_test_path)
        # initialize data loader
        batch_count = len(cnn_dataset) // batch_size
        cnn_dataloader = DataLoader(cnn_dataset, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=None)
        # define model
        cnn_model = MLP(use_cuda=use_cuda)
        cnn_model.to(device)

        # load checkpoint file
        cnn_checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        # model state dict
        cnn_model_state_dict = cnn_checkpoint.pop("model_state_dict")
        cnn_model.load_state_dict(cnn_model_state_dict, strict=True)
        # create empty arrays to hold predicted and true values
        y_true_cnn = np.zeros((len(cnn_dataset),), dtype=np.float32)
        y_pred_cnn = np.zeros((len(cnn_dataset),), dtype=np.float32)
        pdbid_arr = np.zeros((len(cnn_dataset),), dtype=object)
        pred_list = []
        cnn_model.eval()
        with torch.no_grad():
            for batch_ind, batch in enumerate(cnn_dataloader):
                x_batch_cpu, y_batch_cpu = batch
                x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
                bsize = x_batch.shape[0]
                ypred_batch, _ = cnn_model(x_batch[:x_batch.shape[0]])
                ytrue = y_batch_cpu.float().data.numpy()[:,0]
                ypred = ypred_batch.cpu().float().data.numpy()[:,0]
                y_true_cnn[batch_ind*batch_size:batch_ind*batch_size+bsize] = ytrue
                y_pred_cnn[batch_ind*batch_size:batch_ind*batch_size+bsize] = ypred

    if architecture == 'HACNet' or 'gcn0':
        # GCN-0
        gcn0_dataset = GCN_Dataset(gcn0_test_path)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn0_dataloader = DataListLoader(gcn0_dataset, batch_size=7, shuffle=False)
        # define model
        gcn0_model = GeometricDataParallel(GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
        # load checkpoint file
        gcn0_checkpoint = torch.load(gcn0_checkpoint_path, map_location=device)
        # model state dict
        gcn0_model_state_dict = gcn0_checkpoint.pop("model_state_dict")
        gcn0_model.load_state_dict(gcn0_model_state_dict, strict=True)
        test_data_hdf = h5py.File(gcn0_test_path, 'r')
        gcn0_model.eval()
        y_true_gcn0, y_pred_gcn0, pdbid_array = [], [], []
        with torch.no_grad():
            for batch in tqdm(gcn0_dataloader):
                data_list = []
                for dataset in batch:
                    pdbid = dataset[0]
                    pdbid_array.append(pdbid)
                    affinity = test_data_hdf[pdbid].attrs["affinity"].reshape(1,-1)
                    vdw_radii = (test_data_hdf[pdbid].attrs["van_der_waals"].reshape(-1, 1))
                    node_feats = np.concatenate([vdw_radii, test_data_hdf[pdbid][:, 3:22]], axis=1)
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) 
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                    data_list.append(data)
                batch_data = [x for x in data_list if x is not None]
                y_ = gcn0_model(batch_data)
                y = torch.cat([x.y for x in data_list])
                y_true_gcn0.append(y.cpu().data.numpy())
                y_pred_gcn0.append(y_.cpu().data.numpy())
        y_true_gcn0 = np.concatenate(y_true_gcn0).reshape(-1, 1).squeeze(1)
        y_pred_gcn0 = np.concatenate(y_pred_gcn0).reshape(-1, 1).squeeze(1)

    if architecture == 'HACNet' or 'gcn1':
        # GCN-1
        gcn1_dataset = GCN_Dataset(gcn1_test_path)
        # initialize testing data loader
        batch_count = len(gcn0_dataset) // batch_size
        gcn1_dataloader = DataListLoader(gcn1_dataset, batch_size=7, shuffle=False)
        # define model
        gcn1_model = GeometricDataParallel(GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
        # load checkpoint file
        gcn1_checkpoint = torch.load(gcn1_checkpoint_path, map_location=device)
        # model state dict
        gcn1_model_state_dict = gcn1_checkpoint.pop("model_state_dict")
        gcn1_model.load_state_dict(gcn1_model_state_dict, strict=True)
        test_data_hdf = h5py.File(gcn1_test_path, 'r')
        gcn1_model.eval()
        y_true_gcn1, y_pred_gcn1, pdbid_array = [], [], []
        with torch.no_grad():
            for batch in tqdm(gcn1_dataloader):
                data_list = []
                for dataset in batch:
                    pdbid = dataset[0]
                    pdbid_array.append(pdbid)
                    affinity = test_data_hdf[pdbid].attrs["affinity"].reshape(1,-1)
                    vdw_radii = (test_data_hdf[pdbid].attrs["van_der_waals"].reshape(-1, 1))
                    node_feats = np.concatenate([vdw_radii, test_data_hdf[pdbid][:, 3:22]], axis=1)
                    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dataset[1]).float()) 
                    x = torch.from_numpy(node_feats).float()
                    y = torch.FloatTensor(affinity).view(-1, 1)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
                    data_list.append(data)
                batch_data = [x for x in data_list if x is not None]
                y_ = gcn1_model(batch_data)
                y = torch.cat([x.y for x in data_list])
                y_true_gcn1.append(y.cpu().data.numpy())
                y_pred_gcn1.append(y_.cpu().data.numpy())
        y_true_gcn1 = np.concatenate(y_true_gcn1).reshape(-1, 1).squeeze(1)
        y_pred_gcn1 = np.concatenate(y_pred_gcn1).reshape(-1, 1).squeeze(1)

    # compute metrics
    if architecture == 'HACNet':
        y_true = y_true_cnn/3 + y_true_gcn0/3 + y_true_gcn1/3
        y_pred = y_pred_cnn/3 + y_pred_gcn0/3 + y_pred_gcn1/3
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if architecture == 'cnn':
        y_true = y_true_cnn
        y_pred = y_pred_cnn
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if architecture == 'gcn0':
        y_true = y_true_gcn0
        y_pred = y_pred_gcn0
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    if architecture == 'gcn1':
        y_true = y_true_gcn1
        y_pred = y_pred_gcn1
        # define rmse
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # define mae
        mae = mean_absolute_error(y_true, y_pred)
        # define r^2
        r2 = r2_score(y_true, y_pred)
        # define pearson correlation coefficient
        pearson, ppval = pearsonr(y_true, y_pred)
        # define spearman correlation coefficient
        spearman, spval = spearmanr(y_true, y_pred)
        # define mean
        mean = np.mean(y_pred)
        # define standard deviation
        std = np.std(y_pred)

    # print test summary
    print('Test Summary:')
    print('RMSE: %.3f, MAE: %.3f, r^2 score: %.3f, Pearson r: %.3f, Spearman p: %.3f, Mean: %.3f, SD: %.3f' % (rmse, mae, r2, pearson, spearman, mean, std))

    #print scatterplot of true and predicted values
    plt.rcParams["figure.figsize"] = (7,7)
    plt.scatter(y_true, y_pred, color='darkslateblue', edgecolors='black')       # color = firebrick if VANILLA
    plt.xlabel("True", fontsize=20, weight='bold')
    plt.ylabel("Predicted", fontsize=20, weight='bold')
    plt.xticks([0,3,6,9,12], fontsize=18)
    plt.yticks([0,3,6,9,12], fontsize=18)
    plt.xlim(0,14)
    plt.ylim(0,14)
    plt.plot(np.arange(0,15), np.arange(0,15), color='dimgray', linestyle='-',zorder=0, linewidth=2)


""" define a function to predict pkd for a single protein-ligand complex """
def predict_pkd(protein_pdb, ligand_mol2, elements_xml, cnn_params, gcn0_params, gcn1_params, mlp_params, verbose=True): 

  """
  Inputs:
  1) protein_pdb: path to protein pdb file
  2) ligand_mol2: path to ligand mol2 file
  3) verbose: if True, will return pkd and image of protein-ligand complex.
      If False, will return float corresponding to predicted pkd. Default is True
  
  Output:
  1) Prediction of pkd
  2) visual of protein-ligand complex (if verbose == True)
  """

  """ define function to extract pocket from protein pdb """
  def extract_pocket(protein_pdb, ligand_mol2):

    # read in protein pdb file
    protein =  PandasPdb().read_pdb(protein_pdb)

    # define protein atoms dataframe
    protein_atom = protein.df['ATOM'].reset_index(drop=True)

    # create protein atom dictionary
    protein_atom_dict = protein_atom.to_dict('index')

    # define protein heteroatoms dataframe
    protein_hetatm =  protein.df['HETATM'].reset_index(drop=True)

    # create protein heteroatom dictionary
    protein_hetatm_dict = protein_hetatm.to_dict('index')

    # read in ligand mol2 file
    ligand = PandasMol2().read_mol2(ligand_mol2).df

    # define ligand non-H atoms dataframe
    ligand_nonh = ligand[ligand['atom_type'] != 'H'].reset_index(drop=True)

    # create ligand non-H atom dictionary
    ligand_nonh_dict = ligand_nonh.to_dict('index')

    # initialize lists to save IDs for residues and heteroatoms to keep in pocket file
    pocket_residues = []
    pocket_heteroatoms = []

    # for each protein atom:
    for j in range(len(protein_atom_dict)):

      # if residue number is not already saved in list
      if protein_atom_dict[j]['residue_number'] not in pocket_residues:

        # for each ligand non-H atom:
        for i in range(len(ligand_nonh_dict)):

          # if Euclidean distance is within 8 Angstroms:
          if np.sqrt((ligand_nonh_dict[i]['x'] - protein_atom_dict[j]['x_coord'])**2 + (ligand_nonh_dict[i]['y'] - protein_atom_dict[j]['y_coord'])**2 + (ligand_nonh_dict[i]['z'] - protein_atom_dict[j]['z_coord'])**2) <= 8:
            
            # save chain ID, residue number and insertion to list
            pocket_residues.append(str(protein_atom_dict[j]['chain_id'])+'_'+str(protein_atom_dict[j]['residue_number'])+'_'+str(protein_atom_dict[j]['insertion']))            
            
            break
      
    # for each protein heteroatom:
    for k in range(len(protein_hetatm_dict)):

      # if residue number is not already saved in list
      if protein_hetatm_dict[k]['residue_number'] not in pocket_heteroatoms:

        # for each ligand non-H atom:
        for i in range(len(ligand_nonh_dict)):

          # if Euclidean distance is within 8 Angstroms:      
          if np.sqrt((ligand_nonh_dict[i]['x'] - protein_hetatm_dict[k]['x_coord'])**2 + (ligand_nonh_dict[i]['y'] - protein_hetatm_dict[k]['y_coord'])**2 + (ligand_nonh_dict[i]['z'] - protein_hetatm_dict[k]['z_coord'])**2) <= 8:
           
            # save heteroatom ID to list
            pocket_heteroatoms.append(protein_hetatm_dict[k]['residue_number'])

            break

    # initialize list to store atoms that are included in saved residues
    atoms_to_keep = []

    # loop through atom dictionary 
    for k,v in protein_atom_dict.items():

      # if atom is in saved list
      if str(v['chain_id'])+'_'+str(v['residue_number'])+'_'+str(v['insertion']) in pocket_residues:
       
        # append the atom number to new list
        atoms_to_keep.append(v['atom_number'])

    # define the atoms to include in pocket
    residues = protein_atom[(protein_atom['atom_number'].isin(atoms_to_keep))]

    # reset atom number ordering
    residues = residues.reset_index(drop=1)

    # initialize list to store heteroatoms that are included in saved heteroatom IDs
    hetatms_to_keep = []

    # loop through heteroatom dictionary
    for k,v in protein_hetatm_dict.items():

      # if heteroatom is in saved list
      if v['residue_number'] in pocket_heteroatoms and v['residue_name']=='HOH':
       
        # append the heteroatom number to new list
        hetatms_to_keep.append(v['atom_number'])

    # define the heteroatoms to include in pocket file
    heteroatoms = protein_hetatm[(protein_hetatm['atom_number'].isin(hetatms_to_keep))]

    # reset heteroatom number ordering
    heteroatoms = heteroatoms.reset_index(drop=1)

    # initialize biopandas object to write out pocket pdb file
    pred_pocket = PandasPdb()

    # define the atoms and heteroatoms of the object
    pred_pocket.df['ATOM'], pred_pocket.df['HETATM'] = residues, heteroatoms

    # save the created object to a pdb file
    return pred_pocket.to_pdb('/content/pocket.pdb')

  # run function to extract the pocket from protein pdb file
  extract_pocket(protein_pdb, ligand_mol2)

  # initialize a pymol state by first deleting everything
  cmd.delete('all')

  # load in the created pocket pdb file
  cmd.load('/content/pocket.pdb')

  # add hydrogens to the water molecules
  cmd.h_add('sol')

  # save the state as a mol2 file
  cmd.save('/content/pocket.mol2')

  """ define function to calculate charges for pocket mol2 file """
  def add_mol2_charges(pocket_mol2):

    # upload the pocket mol2 file to the ACC2 API
    r=requests.post('http://78.128.250.156/send_files', files={'file[]':open(pocket_mol2,'rb')})

    # obtain ID number for uploaded file
    r_id=list(r.json()['structure_ids'].values())[0]

    # calculate charges using eqeq method
    r_out=requests.get('http://78.128.250.156/calculate_charges?structure_id='+r_id+'&method=eqeq&generate_mol2=true') 

    # save output mol2 file
    open('/content/charged_pocket.mol2', 'wb').write(r_out.content)

  # run function to calculate and add charges to pocket mol2 file
  add_mol2_charges('/content/pocket.mol2')

  # define charged pocket mol2 variable
  pocket_mol2_charged = '/content/charged_pocket.mol2'

  """ define Featurizer class from tfbio source code """
  class Featurizer():

      def __init__(self, atom_codes=None, atom_labels=None,
                  named_properties=None, save_molecule_codes=True,
                  custom_properties=None, smarts_properties=None,
                  smarts_labels=None):
        
          # initialize list to store names of all features in the correct order
          self.FEATURE_NAMES = []

          # validate and process atom codes and labels
          if atom_codes is not None:
              if not isinstance(atom_codes, dict):
                  raise TypeError('Atom codes should be dict, got %s instead'
                                  % type(atom_codes))
                  
              codes = set(atom_codes.values())
              for i in range(len(codes)):
                  if i not in codes:
                      raise ValueError('Incorrect atom code %s' % i)
              
              self.NUM_ATOM_CLASSES = len(codes)
              self.ATOM_CODES = atom_codes
              
              if atom_labels is not None:
                  if len(atom_labels) != self.NUM_ATOM_CLASSES:
                      raise ValueError('Incorrect number of atom labels: '
                                      '%s instead of %s'
                                      % (len(atom_labels), self.NUM_ATOM_CLASSES))
              
              else:
                  atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
             
              self.FEATURE_NAMES += atom_labels

          else:
              self.ATOM_CODES = {}
              metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104)))

              # List of tuples (atomic_num, class_name) with atom types to encode
              atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'),
                  (16, 'S'), (34, 'Se'), ([9, 17, 35, 53], 'halogen'), (metals, 'metal')]
             
              for code, (atom, name) in enumerate(atom_classes):
                  if type(atom) is list:
                      for a in atom:
                          self.ATOM_CODES[a] = code
                 
                  else:
                      self.ATOM_CODES[atom] = code
                  
                  self.FEATURE_NAMES.append(name)
              
              self.NUM_ATOM_CLASSES = len(atom_classes)

          # validate and process named properties
          if named_properties is not None:
              if not isinstance(named_properties, (list, tuple, np.ndarray)):
                  raise TypeError('named_properties must be a list')
             
              allowed_props = [prop for prop in dir(openbabel.pybel.Atom)
                              if not prop.startswith('__')]
              
              for prop_id, prop in enumerate(named_properties):
                  if prop not in allowed_props:
                      raise ValueError(
                          'named_properties must be in pybel.Atom attributes,'
                          ' %s was given at position %s' % (prop_id, prop))
              
              self.NAMED_PROPS = named_properties
        
          else:
              # pybel.Atom properties to save
              self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                  'partialcharge']

          self.FEATURE_NAMES += self.NAMED_PROPS
          
          if not isinstance(save_molecule_codes, bool):
              raise TypeError('save_molecule_codes should be bool, got %s '
                              'instead' % type(save_molecule_codes))
         
          self.save_molecule_codes = save_molecule_codes
         
          if save_molecule_codes:
              # Remember if an atom belongs to the ligand or to the protein
              self.FEATURE_NAMES.append('molcode')

          # process custom callable properties
          self.CALLABLES = []
         
          if custom_properties is not None:
              for i, func in enumerate(custom_properties):
                  if not callable(func):
                      raise TypeError('custom_properties should be list of'
                                      ' callables, got %s instead' % type(func))
                  
                  name = getattr(func, '__name__', '')
                  
                  if name == '':
                      name = 'func%s' % i
                  
                  self.CALLABLES.append(func)
                  
                  self.FEATURE_NAMES.append(name)

          # process SMARTS properties and labels
          if smarts_properties is None:
              # SMARTS definition for other properties
              self.SMARTS = [
                  '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                  '[a]',
                  '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                  '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                  '[r]']
              
              smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                              'ring']
          
          elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
              raise TypeError('smarts_properties must be a list')
          
          else:
              self.SMARTS = smarts_properties

          if smarts_labels is not None:
              if len(smarts_labels) != len(self.SMARTS):
                  raise ValueError('Incorrect number of SMARTS labels: %s'
                                  ' instead of %s'
                                  % (len(smarts_labels), len(self.SMARTS)))
          
          else:
              smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

          # Compile SMARTS patterns for matching
          self.compile_smarts()

          self.FEATURE_NAMES += smarts_labels

      # define function to compile SMARTS patterns for efficient matching
      def compile_smarts(self):
          self.__PATTERNS = []
          
          for smarts in self.SMARTS:
              self.__PATTERNS.append(openbabel.pybel.Smarts(smarts))

      # define function to encode the atomic number using one-hot encoding
      def encode_num(self, atomic_num):

          if not isinstance(atomic_num, int):
              raise TypeError('Atomic number must be int, %s was given'
                              % type(atomic_num))

          encoding = np.zeros(self.NUM_ATOM_CLASSES)
          
          try:
              encoding[self.ATOM_CODES[atomic_num]] = 1.0
          
          except:
              pass
          
          return encoding

      # define function to find substructures in the molecule that match the SMARTS patterns
      def find_smarts(self, molecule):

          if not isinstance(molecule, openbabel.pybel.Molecule):
              raise TypeError('molecule must be pybel.Molecule object, %s was given'
                              % type(molecule))

          features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

          for (pattern_id, pattern) in enumerate(self.__PATTERNS):
              atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                          dtype=int) - 1

              features[atoms_with_prop, pattern_id] = 1.0

          return features

      # define function to extract the features from the molecule
      def get_features(self, molecule, molcode=None):

          if not isinstance(molecule, openbabel.pybel.Molecule):
              raise TypeError('molecule must be pybel.Molecule object,'
                              ' %s was given' % type(molecule))
         
          if molcode is None:
              if self.save_molecule_codes is True:
                  raise ValueError('save_molecule_codes is set to True,'
                                  ' you must specify code for the molecule')
          
          elif not isinstance(molcode, (float, int)):
              raise TypeError('motlype must be float, %s was given'
                              % type(molcode))

          coords = []
          features = []
          heavy_atoms = []

          for i, atom in enumerate(molecule):
             
              # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
              if atom.atomicnum > 1:
                  heavy_atoms.append(i)
                  
                  coords.append(atom.coords)
                  
                  features.append(np.concatenate((
                      self.encode_num(atom.atomicnum),
                      [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                      [func(atom) for func in self.CALLABLES])))

          coords = np.array(coords, dtype=np.float32)
          
          features = np.array(features, dtype=np.float32)
          
          if self.save_molecule_codes:
              features = np.hstack((features,
                                      molcode * np.ones((len(features), 1))))
         
          features = np.hstack([features,
                                  self.find_smarts(molecule)[heavy_atoms]])

          if np.isnan(features).any():
              raise RuntimeError('Got NaN when calculating features')

          return coords, features

      # define function to save the Featurizer to a pickle file
      def to_pickle(self, fname='featurizer.pkl'):
          
          # patterns can't be pickled, we need to temporarily remove them
          patterns = self.__PATTERNS[:]
          
          del self.__PATTERNS
          
          try:
              with open(fname, 'wb') as f:
                  pickle.dump(self, f)
         
          finally:
              self.__PATTERNS = patterns[:]

      @staticmethod
      def from_pickle(fname):
          with open(fname, 'rb') as f:
              featurizer = pickle.load(f)
          
          featurizer.compile_smarts()
         
          return featurizer

  """ define function to featurize mol2 files for use in HAC-Net """
  def prepare_data(pocket_mol2_charged, ligand_mol2, elements_xml):

      # define function to extract features from the binding pocket mol2 file 
      def __get_pocket():
            # define pocket from input pocket_mol2 file
            pocket = next(openbabel.pybel.readfile('mol2', pocket_mol2_charged))
            
            # obtain pocket coordinates and features
            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)

            # obtain Van der Waals radii of pocket atoms
            pocket_vdw = parse_mol_vdw(mol=pocket, element_dict=element_dict)
            
            yield (pocket_coords, pocket_features, pocket_vdw)

      # define function to extract information from elements_xml file
      def parse_element_description(xml_file):
          # initialize dictionary to store element chemical information
          element_info_dict = {}

          # parse and define elements_xml file
          element_info_xml = ET.parse(xml_file)

          # for each element in file
          for element in element_info_xml.iter():
              
              # if 'comment' is keys of element attributes
              if "comment" in element.attrib.keys():

                  #continue
                  continue
              
              # if 'comment' not in keys of element attributes
              else:
                  # save the attribute to the dictionary value to 'number
                  element_info_dict[int(element.attrib["number"])] = element.attrib
          
          # return dictionary containing element information
          return element_info_dict

      # define function to create a list of van der Waals radii for a molecule
      def parse_mol_vdw(mol, element_dict):

        # initialize list to store Van der Waals radii
          vdw_list = []
          
          # for each atom in molecule
          for atom in mol.atoms:

              # if the atom is not Hydrogen
              if int(atom.atomicnum)>=2:

                  # append the Van der Waals radius to the list
                  vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))
         
         # return the Van der Waals list as a numpy array
          return np.asarray(vdw_list)

      # read in elements_xml and store important information in dictionary
      element_dict = parse_element_description(elements_xml)

      # define featurizer object
      featurizer = Featurizer()

      # define object for getting features of pocket
      pocket_generator = __get_pocket()

      # read ligand file using pybel
      ligand = next(openbabel.pybel.readfile('mol2', ligand_mol2))

      # extract coordinates, 19 features, and Van der Waals radii from pocket atoms
      pocket_coords, pocket_features, pocket_vdw = next(pocket_generator)

      # extract coordinates, and 19 features from ligand atoms
      ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)

      # extract Van der Waals radii from ligand atoms
      ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)
      
      # define centroid to be the center of the ligand
      centroid = ligand_coords.mean(axis=0)

      # normalize ligand coordinates with respect to centroid
      ligand_coords -= centroid

      #normalize pocket coordinates with respect to centroid
      pocket_coords -= centroid

      # assemble the features into one large numpy array where rows are heavy atoms, and columns are coordinates and features
      data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
          np.concatenate((ligand_features, pocket_features))), axis=1)
      
      # concatenate van der Waals radii into one numpy array
      vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))

      # return properly formatted coordinates, features, and Van der Waals radii
      return data, vdw_radii

  # prepare data from input files
  prep_data, prep_vdw = prepare_data(pocket_mol2_charged, ligand_mol2, elements_xml)

  # define function to voxelize input data for use in 3D-CNN component
  def voxelize_one_vox_per_atom(xyz_array, feat, vol_dim):
    
    # initialize volume
    vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

    # get coordinates of center of x axis
    xmid = (min(xyz_array[:, 0]) + max(xyz_array[:, 0])) / 2

    # get coordinates of center of y axis
    ymid = (min(xyz_array[:, 1]) + max(xyz_array[:, 1])) / 2

    # get coordinates of center of z axis
    zmid = (min(xyz_array[:, 2]) + max(xyz_array[:, 2])) / 2

    # define minimum x coordinate
    xmin = xmid - (48 / 2)

    # define minimum y coordinate
    ymin = ymid - (48 / 2)

    # define minimum z coordinate
    zmin = zmid - (48 / 2)

    # define maximum x coordinate
    xmax = xmid + (48 / 2)

    # define maximum y coordinate
    ymax = ymid + (48 / 2)

    # define maximum z coordinate
    zmax = zmid + (48 / 2)

    # assign each atom to the voxel that contains its center
    for ind in range(xyz_array.shape[0]):

        # define x coordinate
        x = xyz_array[ind, 0]

        # define y coordinate
        y = xyz_array[ind, 1]

        # define z coordinate
        z = xyz_array[ind, 2]

        # if atom is not within the voxel grid
        if x < xmin or x >= xmax or y < ymin or y >= ymax or z < zmin or z >= zmax:

            # continue
            continue

        # determine x-axis index of voxel to contain atom
        cx = math.floor((x - xmin) / (xmax - xmin) * (vol_dim[1]))

        # determine y-axis index of voxel to contain atom
        cy = math.floor((y - ymin) / (ymax - ymin) * (vol_dim[2]))

        # determine z-axis index of voxel to contain atom
        cz = math.floor((z - zmin) / (zmax - zmin) * (vol_dim[3]))

        # add to each voxel the features of the assigned atom
        vol_data[:, cx, cy, cz] += feat[ind, :]

    # return voxelized data
    return vol_data

  # prepare voxelized data
  prep_data_vox = voxelize_one_vox_per_atom(prep_data[:, 0:3], prep_data[:, 3:], [prep_data.shape[1]-3, 48, 48, 48])

  # add an additional axis and convert to a tensor
  prep_data_vox = torch.tensor(prep_data_vox[np.newaxis,...])


  """ define a function to extract flattened features from trained 3D-CNN """
  def extract_features(checkpoint_path):

      # set CUDA for PyTorch
      use_cuda = torch.cuda.is_available()
      cuda_count = torch.cuda.device_count()
      
      # if using GPU
      if use_cuda:
          device_name = "cuda:0"
          device = torch.device(device_name)
          torch.cuda.set_device(int(device_name.split(':')[1]))
     
      # if not using GPU
      else:
          # use CPU
          device = torch.device("cpu")

      # define model
      model = CNN(use_cuda=use_cuda)     
      model.to(device)
      
      # load checkpoint file
      checkpoint = torch.load(checkpoint_path, map_location=device)
      
      # prepare model state dict from CNN parameter file
      model_state_dict = checkpoint.pop("model_state_dict")
  
      # load state dict
      model.load_state_dict(model_state_dict, strict=True)

      # put model in evaluation mode
      model.eval()
      
      with torch.no_grad():

          # define flattened features
          x_batch_cpu = prep_data_vox
          x_batch = x_batch_cpu.to(device)
          _, flat_feat_batch = model(x_batch)
          flatfeat = flat_feat_batch.cpu().data.numpy()

      # return flattened features
      return flatfeat
  
  # define extracted features for training MLP
  flat_feat = extract_features(checkpoint_path = cnn_params)

  """ define function to perform forward pass and return prediction """
  def test_HACNet(cnn_params, gcn0_params, gcn1_params):

      # set CUDA for PyTorch
      use_cuda = torch.cuda.is_available()
      cuda_count = torch.cuda.device_count()
      device_name = "cuda:0"
      
      # if using GPU
      if use_cuda:
          device = torch.device(device_name)
          torch.cuda.set_device(int(device_name.split(':')[1]))
      
      # if not using GPU
      else:

          # use CPU
          device = torch.device('cpu')  

      # define 3D-CNN (MLP) model
      cnn_model = Model_Linear(use_cuda=use_cuda)
      cnn_model.to(device)
      
      # load checkpoint file
      cnn_checkpoint = torch.load(cnn_params, map_location=device)
     
      # load in parameters and fill model state dict
      cnn_model_state_dict = cnn_checkpoint.pop("model_state_dict")
      cnn_model.load_state_dict(cnn_model_state_dict, strict=False)

      # put model in evaluation mode
      cnn_model.eval()
      
      with torch.no_grad():

          # obtain 3D-CNN (MLP) prediction from flattened feature input
          x_batch_cpu = torch.tensor(flat_feat)
          x_batch = x_batch_cpu.to(device)
          ypred_batch, _ = cnn_model(x_batch)
          ypred = ypred_batch.cpu().float().data.numpy()[:,0]
          y_pred_cnn = ypred

      # define first GCN model (GCN0)
      gcn0_model = GeometricDataParallel(MP_GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
      
      # load checkpoint file
      gcn0_checkpoint = torch.load(gcn0_params, map_location=device)
      
      # fill model state dict with parameters
      gcn0_model_state_dict = gcn0_checkpoint.pop("model_state_dict")
      gcn0_model.load_state_dict(gcn0_model_state_dict, strict=False)

      # put model in evaluation mode
      gcn0_model.eval()

      # initialize list to store predictions
      y_pred_gcn0 = []
     
      with torch.no_grad():

        # obtain predictions from input data
        data_list = []
        vdw_radii = prep_vdw.reshape(-1, 1)
        node_feats = np.concatenate([vdw_radii,prep_data[:, 3:22]], axis=1)
        coords=prep_data[:,0:3]
        dists=pairwise_distances(coords, metric='euclidean')
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float()) 
        x = torch.from_numpy(node_feats).float()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1))
        data_list.append(data)
        batch_data = [x for x in data_list if x is not None]
        y_ = gcn0_model(batch_data).cpu().data.numpy()

      # define predictions of first GCN
      y_pred_gcn0 = np.concatenate(y_).reshape(-1, 1).squeeze(1)

      # define second GCN model (GCN1)
      gcn1_model = GeometricDataParallel(MP_GCN(in_channels=20, gather_width=128, prop_iter=4, dist_cutoff=3.5)).float()
    
      # load checkpoint file
      gcn1_checkpoint = torch.load(gcn1_params, map_location=device)
      
      # fill model state dict with parameters
      gcn1_model_state_dict = gcn1_checkpoint.pop("model_state_dict")
      gcn1_model.load_state_dict(gcn1_model_state_dict, strict=False)

      # put second GCN in evaluation mode
      gcn1_model.eval()

      # initialize list to store predictions
      y_pred_gcn1 = []
      
      with torch.no_grad():

        # obtain predictions from input data
        data_list = []
        vdw_radii = prep_vdw.reshape(-1, 1)
        node_feats = np.concatenate([vdw_radii,prep_data[:, 3:22]], axis=1)
        coords=prep_data[:,0:3]
        dists=pairwise_distances(coords, metric='euclidean')
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float()) 
        x = torch.from_numpy(node_feats).float()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1))
        data_list.append(data)
        batch_data = [x for x in data_list if x is not None]
        y_ = gcn1_model(batch_data).cpu().data.numpy()

      # define predictions of second GCN
      y_pred_gcn1 = np.concatenate(y_).reshape(-1, 1).squeeze(1)

      # in the very rare case that the 3D-CNN provides an unreasonable prediction (~ 1 in 1000 PDBbind complexes):
      if abs(y_pred_cnn) >= 50:

        # return the average prediction of the two GCN models
        y_pred = y_pred_gcn0/2 + y_pred_gcn1/2

      # if 3D-CNN provides a reasonable prediction (over 99.9% of time)
      else:

        # return the prediction of binding affinity from complete model
        y_pred = y_pred_cnn/3 + y_pred_gcn0/3 + y_pred_gcn1/3

      # return prediction
      return y_pred[0]

  # define predicted binding affinity (pKd)
  pkd = test_HACNet(cnn_params = mlp_params, 
            gcn0_params = gcn0_params, 
            gcn1_params = gcn1_params)

  # remove intermediate files
  os.remove('/content/pocket.pdb')
  os.remove('/content/pocket.mol2')
  os.remove('/content/charged_pocket.mol2')

  # if verbose is False
  if verbose ==False:

    # return only pKd
    return round(pkd, 3)

  # if verbose is True
  if verbose == True:

    # return pkd and visual of protein-ligand complex
    print('Predicted binding affinity:', str(round(pkd, 3))+' pKd')

    # initialize PyMOL session by deleting everything
    cmd.delete('all')

    # load in ligand twice for representing it visually two different ways
    cmd.load(ligand_mol2, 'ligand')
    cmd.load(ligand_mol2, 'ligand2')

    # load in initial protein file
    cmd.load(protein_pdb, 'pocket')

    # remove waters
    cmd.remove('sol')

    # color the ligand and protein
    cmd.color('lime', 'ligand2')
    cmd.color('violetpurple', 'pocket')
    cmd.color('limon', 'ligand')

    # show protein as cartoon
    cmd.show_as('cartoon', 'pocket')

    # set size of spheres to represent ligand atoms
    cmd.set('sphere_scale', 0.3)

    # show ligand as surface
    cmd.show_as('surface', 'ligand2')

    # set transparencies
    cmd.set('transparency', 0.55, 'ligand2')
    cmd.set('transparency', 0.2, 'pocket')

    # zoom in on the ligand
    cmd.zoom('ligand', 9)

    # save session as image
    cmd.png('image.png')

    # display the image
    display(Image('image.png', unconfined=True))
