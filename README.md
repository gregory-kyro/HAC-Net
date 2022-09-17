 # HAC-Net: A Hybrid Attention-Based Convolutional Neural Network for Predicting Protein-Ligand Binding Affinity

Applying deep learning concepts from image detection and graph theory has greatly advanced protein-ligand binding affinity prediction, a challenge with enormous ramifications for both protein engineering and drug development. We build upon these advances by designing a novel architecture which demonstrates unmatched performance on the 2016 PDBbind core set, the most widely-recognized benchmark in the field. HAC-Net (Hybrid Attention-based Convolutional Neural Network) consists of a 3D Convolutional Neural Network (3D-CNN) utilizing channel-wise attention and two Message-Passing Graph Convolutional Networks (MP-GCNs) utilizing attention-based aggregation of node features. We rigorously assess the quality of our model with numerous different train-test splits generated from the PDBbind database. Additionally, we envision that this state-of-the-art model can be extended to a broad range of supervised learning problems related to structure-based biomolecular property prediction. 

Table. Comparison to other high-performing models for predicting protein-ligand binding affinity on crystal structures of the PDBbind 2016 core set. The best value for each metric is shown as bold.
![Picture3](https://user-images.githubusercontent.com/98780179/190861777-fdc5889f-0c8d-4876-9406-a8e7b77ffc83.png)




Figure. Correlation scatter plots depicting performance of HAC-Net on crystal structures of the PDBbind 2016 core set. r2, Spearman ρ, and Pearson r are shown on plots. Performance is shown for the A) hybrid model, B) 3D-CNN, C) one of the two MP-GCNs, and D) the other MP-GCN. E) Basic scheme of hybrid model architecture.
![Picture4](https://user-images.githubusercontent.com/98780179/190861822-40bd2ab2-91ca-4cf2-9659-d28f75438531.png)
