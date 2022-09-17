 # HAC-Net: A Hybrid Attention-Based Convolutional Neural Network for Predicting Protein-Ligand Binding Affinity

Applying deep learning concepts from image detection and graph theory has greatly advanced protein-ligand binding affinity prediction, a challenge with enormous ramifications for both protein engineering and drug development. We build upon these advances by designing a novel architecture which demonstrates unmatched performance on the 2016 PDBbind core set, the most widely-recognized benchmark in the field. HAC-Net (Hybrid Attention-based Convolutional Neural Network) consists of a 3D Convolutional Neural Network (3D-CNN) utilizing channel-wise attention and two Message-Passing Graph Convolutional Networks (MP-GCNs) utilizing attention-based aggregation of node features. We rigorously assess the quality of our model with numerous different train-test splits generated from the PDBbind database. Additionally, we envision that this state-of-the-art model can be extended to a broad range of supervised learning problems related to structure-based biomolecular property prediction. 

## 3D-CNN
![Picture1](https://user-images.githubusercontent.com/98780179/190861629-6536a183-a82e-4360-88be-14b94ad8de12.png)

## MP-GCN!
![Picture2](https://user-images.githubusercontent.com/98780179/190861694-52cad368-eeaf-4b50-a3f2-2ed4c5d31cc2.png)
