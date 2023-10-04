![hacnet_logo-removebg-preview (2)](https://user-images.githubusercontent.com/98780179/198727732-de8a6370-0086-4d1e-a827-e7de432f2716.png)

# HAC-Net: A Hybrid Attention-Based Convolutional Neural Network for Highly Accurate Protein-Ligand Binding Affinity Prediction

[![image](https://img.shields.io/pypi/v/HACNet.svg)](https://pypi.org/project/HACNet/)

<a target="_blank" href="https://colab.research.google.com/github/gregory-kyro/HAC-Net/blob/main/HACNet.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Summary
Applying deep learning concepts from image detection and graph theory has greatly advanced protein-ligand binding affinity prediction, a challenge with enormous ramifications for both drug discovery and protein engineering. We build upon these advances by designing a novel deep learning architecture consisting of a 3-dimensional convolutional neural network utilizing channel-wise attention and two graph convolutional networks utilizing attention-based aggregation of node features. HAC-Net (Hybrid Attention-Based Convolutional Neural Network) obtains state-of-the-art results on the PDBbind v.2016 core set, the most widely recognized benchmark in the field. We extensively assess the generalizability of our model using multiple train-test splits, each of which maximizes differences between either protein structures, protein sequences, or ligand extended-connectivity fingerprints. Furthermore, we perform 10-fold cross-validation with a similarity cutoff between SMILES strings of ligands in the training and test sets, and also evaluate the performance of HAC-Net on lower-quality data. We envision that this model can be extended to a broad range of supervised learning problems related to structure-based biomolecular property prediction.

## Overview of Model

![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/98780179/206188596-032a4f78-4af1-48e8-8cc0-f1800587ddab.gif)

HAC-Net (Hybrid Attention-Based Convolutional Neural Network) is a novel deep learning architecture for protein-ligand binding affinity prediction consisting of a 3D-CNN utilizing channel-wise attention and two GCNs utilizing attention-based aggregation of node features. This combination achieves an optimal balance between the superior performance of our GCNs and the complementary learning style of our 3D-CNN. Furthermore, the inclusion of two architecturally-identical GCNs mitigates noise resulting from the inherently-stochastic nature of the training process. By incorporating multiple forms of attention with advanced concepts from CNN and GCN architectural design, we are able to demonstrate state-of-the-art performance on the PDBbind benchmark for protein-ligand binding affinity prediction, as well the ability to generalize to complexes unlike those used for training.

## Demo Video of HACNet Python Package
https://user-images.githubusercontent.com/98780179/236011365-6597f75e-8b19-4f9e-93d6-944875227a22.mp4

## Tutorial Notebook Using HACNet Python Package
https://colab.research.google.com/github/gregory-kyro/HAC-Net/blob/main/HACNet.ipynb

## Important Files and Notebooks
All of the:
1) HDF files used for training, validation and testing
2) NPY files containing 3D-CNN extracted features
3) PT files containing model parameters
4) IPYNB files of tutorial notebooks for training and testing

can be found at: https://drive.google.com/drive/folders/1yB2voUxwzhrQRh0JXnOD3BzY8ZQrbgUK?usp=sharing

## Associated Journal Article
https://pubs.acs.org/doi/10.1021/acs.jcim.3c00251

## Associated Preprint
https://arxiv.org/abs/2212.12440

## Python Package
https://pypi.org/project/HACNet/

in order to install the HACNet package, simply run:

```pip install HACNet```

## Contact
Please feel free to reach out to us through either of the following emails if you have any questions or need any additional files:

gregory.kyro@yale.edu

rafi.brent@yale.edu
