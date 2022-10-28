![hacnet_logo-removebg-preview (2)](https://user-images.githubusercontent.com/98780179/198727732-de8a6370-0086-4d1e-a827-e7de432f2716.png)

# HAC-Net: A Hybrid Attention-Based Convolutional Neural Network for Predicting Protein-Ligand Binding Affinity

## Summary
Applying deep learning concepts from image detection and graph theory has greatly advanced protein-ligand binding affinity prediction, a challenge with enormous ramifications for both protein engineering and drug development. We build upon these advances by designing a novel architecture which demonstrates unmatched performance on the 2016 PDBbind core set, the most widely-recognized benchmark in the field. HAC-Net (**H**ybrid **A**ttention-based **C**onvolutional Neural **Net**work) consists of a 3D Convolutional Neural Network (3D-CNN) utilizing channel-wise attention and two Message-Passing Graph Convolutional Networks (MP-GCNs) utilizing attention-based aggregation of node features. We rigorously assess the quality of our model with numerous different train-test splits generated from the PDBbind database. Additionally, we envision that this state-of-the-art model can be extended to a broad range of supervised learning problems related to structure-based biomolecular property prediction. 

## Overview of Model
![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/98780179/198733695-16f18f58-b627-4df8-845a-4291b05b5c75.gif)

HAC-Net is a deep learning model composed of one ResNet-inspired 3D-CNN and two identically structured but independently trained message-passing GCNs (MP-GCNs). The model takes as the inputs protein and ligand structural files and outputs a prediction of the binding affinity between the inputs. While previous machine learning models for binding affinity prediction have focused primarily on working with either voxelized or graph-based representations of molecular complexes, recent work has demonstrated that combining models which operate on differently-structured data can increase performance compared to either one in isolation. Utilizing a similar approach, we designed a hybrid model consisting of one 3D-CNN and two independently-trained MP-GCNs. This combination achieves an optimal balance between the superior performance of our MP-GCN and the value of including a disparate feature extraction method provided by the 3D-CNN. Furthermore, the inclusion of two analogous GCNs mitigates noise due to the inherently-stochastic nature of the training process. Our hybrid model involves no additional trainings, and instead averages the predictions of the three independent models to generate a final output.


## Contact
Please feel free to reach out to us through either of the following emails if you have any questions or need any additional files:

gregory.kyro@yale.edu

rafi.brent@yale.edu
