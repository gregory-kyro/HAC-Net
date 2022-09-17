 # HAC-Net: A Hybrid Attention-Based Convolutional Neural Network for Predicting Protein-Ligand Binding Affinity

## Summary
Applying deep learning concepts from image detection and graph theory has greatly advanced protein-ligand binding affinity prediction, a challenge with enormous ramifications for both protein engineering and drug development. We build upon these advances by designing a novel architecture which demonstrates unmatched performance on the 2016 PDBbind core set, the most widely-recognized benchmark in the field. HAC-Net (Hybrid Attention-based Convolutional Neural Network) consists of a 3D Convolutional Neural Network (3D-CNN) utilizing channel-wise attention and two Message-Passing Graph Convolutional Networks (MP-GCNs) utilizing attention-based aggregation of node features. We rigorously assess the quality of our model with numerous different train-test splits generated from the PDBbind database. Additionally, we envision that this state-of-the-art model can be extended to a broad range of supervised learning problems related to structure-based biomolecular property prediction. 

## Overview of Model
HAC-Net (Hybrid Attention-based Convolutional Neural Network) is a deep learning model composed of one ResNet-inspired 3D-CNN and two identically structured but independently trained message-passing GCNs (MP-GCNs). The model takes as the inputs protein and ligand structural files and outputs a prediction of the binding affinity between the inputs. 

## 3D Convolutional Neural Network
Protein and ligand atoms are first embedded into a 3D spatial grid, each voxel of which is a channel corresponding to a given atomic feature element. In the case of HAC-Net, the input volume dimensions are 48×48×48@19, where 48 corresponds to the voxel grid size, and 19 corresponds to the number of channels. This information is presented to the model as a 4D array. We utilize the atomic feature set first presented by Pafnucy:   
- 9 bits (0 or 1) encoding atom types: B, C, N, O, P, S, Se, halogen and metal
-	1 integer (1, 2, or 3) for atom hybridization
- 1 integer counting the numbers of bonds with other heavy atoms 
- 1 integer counting the numbers of bonds with other heteroatoms
- 5 bits (0 or 1) encoding hydrophobic, aromatic, acceptor, donor and ring
- 1 float for partial charge
- 1 integer (-1 or 1) to distinguish between protein and ligand, respectively

While our model makes use of multiple architectural elements (Fig. 1), the most fundamental building block is the convolutional layer. Intuitively, this component creates a linear combination of all channel values in the spatial neighborhood of a given voxel, then propagates the resulting scalar to a corresponding spatial index in the output array. The coefficients for this linear combination constitute a filter, which is applied uniformly across the input voxels, updating its weights throughout the training. One filter will therefore generate a 3D output array. By applying multiple independent filters to a given input, the channel dimension of the feature map output can be modulated, where each filter corresponds to a channel of the output. 

A 3D convolution is applied over a cubic input signal of size Lx×Ly×Lz@C to generate an output of size L’x×L’y×L’z@C’. We are able to modulate the size of the feature map output by manipulating padding and stride parameters applied to the convolution, where padding refers to inserting zeroes around the input array to increase the size, and stride refers to the step size of the filter upon each convolution.

A major advance in CNN architectural design was achieved by Resnet, incorporating skip connections between layers to address the vanishing gradient problem using identity mapping. In this instance, the input and output of a convolutional layer are added together, and the subsequent convolutional layer will operate on their sum. Architectures containing residual layers are more easily optimized than those relying primarily on standard convolutions, allowing for the training of significantly deeper neural networks which obtain greatly improved results on standard image-recognition benchmarks.

The key component of our 3D-CNN architecture is the SE block, which begins with a standard convolution of the type described above. The values at each channel are averaged across all spatial dimensions, yielding a one-dimensional vector with each index corresponding to a channel. Next, the “squeezed” vector is passed through a pair of linear layers with ReLU activation after the first and Sigmoid activation after the second, producing a transformed array of the same length as the original. Finally, each element of this array is used as a multiplicative factor for the corresponding channel of the original 4D output of the convolution. In this way, the model learns to optimally weight the various features based on a transformation of their collective average values, which can be regarded as a self-attention mechanism on the channels.
           
Our 3D-CNN procedure consists of both a feature extraction protocol and subsequent linear layer optimization for final predictions. 

![Picture1](https://user-images.githubusercontent.com/98780179/190867872-9b0f94ea-8458-4fdc-b798-631e5408f674.png)

The voxelized protein and ligand structural data is first expanded from size 48×48×48@19 to 48×48×48@64, and then passed to an SE block with filter size 9×9×9@64, where the size of the data is reduced to 24×24×24@64. Following the first SE block, the data is then passed to two residual blocks of size 7×7×7@64. The data is then inputted into another SE block of size 7×7×7@128, and outputted with size 8×8×8@128. We then apply max pooling with a filter size of 2×2×2, which divides the spatial grid into sub-grids of size 2×2×2 and propagates the maximal value of each, reducing the size of the data to 4×4×4@128. The data is then inputted into a third SE block of size 5×5×5@256, changing the size of the data to 2×2×2@256. Lastly, the data is flattened into a vector of size 2048 and passed to a linear layer with ReLU activation and batch normalization of size 2048×100, then to a final linear layer of size 100×1, resulting in a binding affinity prediction. 

After a full training of the 3D-CNN model is complete, we separately train a pair of linear layers identical to those used in the model with the flattened features generated by the convolutional layers as training data. This notably improves performance due to the fact that the linear layers account for only 1.9% of the total parameters in the 3D-CNN (as compared with 59.5% in the GCN), causing the initial learning to be driven primarily by the parameters for the convolutional layers. Therefore, independently training the linear layers on the extracted features enables them to adapt more precisely to the outputs of the convolutional layers.

## Message-Passing Graph Convolutional Network

The GCN interacts with the input data in a fundamentally different manner than the 3D-CNN. Rather than using a voxel representation of atoms, the protein-ligand complex is represented as a graph, where nodes correspond to atoms and edges are pathways for information transfer between the nodes, defined by a distance cutoff. In the case of HAC-Net, we use a distance cutoff of 3.5 Å; this is a common donor-acceptor distance cutoff for energetically significant hydrogen bonds in proteins, and thus will contain most of the meaningful interatomic interactions. Therefore, atoms whose centers are within 3.5 Å will have an edge connecting them, allowing messages to be passed between them during propagation. “Messages” in this case refers to atomic features of neighboring nodes that will be passed to the central node and used to update the features of the central node. For the MP-GCN, we utilize the Pafnucy feature set with the addition of Van der Waals radius, for a total of 20 atomic features. 

MP-GCNs are a broad class of networks which iteratively update node features according to three general steps: message creation, aggregation, and feature updating. In message creation, a dimensionality-preserving linear transformation is applied to each set of node features. Once created, messages of neighboring nodes are aggregated according to a specified algorithm. In the case of HAC-Net, we apply a soft attention mechanism similar to that of GGS-NNs, with the important distinction that we incorporate the operation into the aggregation step, while the original use is for aggregating all node features into a set of graph-level features. After message creation and aggregation, the current node features are updated by combining the original node features (pre-message creation) with the node features after aggregation. 

The defining characteristic of a GGS-NN is the use of a Gated Recurrent Unit (GRU) Recurrent Neural Network (RNN) as the update function. We utilize a simplified GRU for updating node features.

Our model performs four iterations of message passing; the outputs of the fourth GRU iteration are combined with the initial set of node features (pre-message-passing), a method presented in FAST (Fusion models for Atomic and molecular Structures), which we refer to as “asymmetric attentional aggregation”. The resulting vector contains graph-level information, and is passed through a final set of three linear layers (128×85, 85×64, and 64×1) with ReLU activation to generate a binding affinity prediction.

![Picture2](https://user-images.githubusercontent.com/98780179/190868102-66522fe5-f7bf-44d7-bd02-f70dda7dff6a.png)

## Hybrid model

While previous machine learning models for binding affinity prediction have focused primarily on working with either voxelized or graph-based representations of molecular complexes, recent work has demonstrated that combining models which operate on differently-structured data can increase performance compared to either one in isolation. Utilizing a similar approach, we designed a hybrid model consisting of one 3D-CNN and two independently-trained MP-GCNs. This combination achieves an optimal balance between the superior performance of our MP-GCN and the value of including a disparate feature extraction method provided by the 3D-CNN. Furthermore, the inclusion of two analogous GCNs mitigates noise due to the inherently-stochastic nature of the training process. Our hybrid model involves no additional trainings, and instead averages the predictions of the three independent models to generate a final output.

## Performance
 
The 2016 PDBbind core set, compiled from the 2016 Comparative Assessment of Scoring Functions (CASF-2016) core set, is the most widely-reported benchmark for protein-ligand binding affinity prediction. This is a subset of the 2016 PDBbind refined set, members of which must satisfy an extensive list of quality requirements. Additionally, the core set members were chosen from a wide distribution of structural clusters and binding affinities, serving as the current gold standard of the field. We therefore test and report results on the core set to directly compare the performance of HAC-Net to that of the highest-performing models in the literature . We obtain results competitive with the highest-performing models in the literature prior to incorporating attention into the architectures. By introducing multiple forms of attention, specifically SE blocks into the 3D-CNN and node-level attentional aggregation into the MP-GCNs, we achieve greater performance.

<h1 align="center">Table. Comparison to other high-performing models for predicting protein-ligand binding affinity on crystal structures of the PDBbind 2016 core set. The best value for each metric is shown as bold.</h1>

![Picture3](https://user-images.githubusercontent.com/98780179/190868153-7c651f72-6cba-442d-8b89-bd9423842765.png)

## Attention-Based Implementations

The results presented here suggest that deep learning models for binding affinity can benefit significantly from incorporating forms of attention into their feature-extraction protocols. This is conveyed in the 3D-CNN with SE convolution blocks providing a self-attention mechanism on the channels, and in the MP-GCN with a soft attention mechanism applied to the aggregation of messages from neighboring nodes.  To demonstrate this point, we independently trained and tested an analogous hybrid model without these multiple forms of attention, and compared the performance to that of HAC-Net. We refer to the 3D-CNN without squeeze and excitation applied to the convolutional procedure as the “vanilla 3D-CNN” and to the MP-GCN without soft attention applied to the aggregation step as the “vanilla MP-GCN”.

![Picture5](https://user-images.githubusercontent.com/98780179/190868298-2f652873-9b87-479c-81b1-0e09ce9d2b8e.png)

![Picture4](https://user-images.githubusercontent.com/98780179/190868225-e253fab9-7ecd-4732-a28e-caa0556b4455.png)
