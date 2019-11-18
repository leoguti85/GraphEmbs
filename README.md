# GraphEmbs
## Unsupervised Network Embeddings With Node Identity Awareness
### (Before: Unsupervised Network Embeddings for Graph Visualization, Clustering and Classification)


In this work we provide an unsupervised approach to learn embedding representations for a collection of graphs defined on the same set of nodes, so that it can be used in numerous graph mining tasks. By using an unsupervised neural network approach [1] on input graphs, we aim to capture the underlying distribution of the data in order to discriminate between different class of networks. Our method is assessed empirically on synthetic and real life datasets and evaluated in three different tasks: graph clustering, visualization and classification. 

The original paper can be found [here](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0197-1 " journal paper")

This code was tested on Debian GNU/Linux 8.11 (jessie), python 3.5.2

## Usage
The script run_synthetic_graphs.py will allow to reproduce the visualization and clustering results reported in the above paper.
### Example
Learning graph embeddings for Erdős–Rényi networks generated from different parameters.
Visualizatiton is done with Multi-scale SNE tool [2]:

```
python run_synthetic_graphs.py -f ER -opc 0
```
![emb](https://github.com/leoguti85/GraphEmbs/blob/master/images/ER.png)

*You can plot the graph embeddings faster with t-SNE by uncommenting the line data.visualize_tsne()*

Clustering graph embeddings for Erdős–Rényi networks in the **embedding space**, as well as the evaluation of its performance with the normalized mutual information metric:

```
python run_synthetic_graphs.py -f ER -opc 1
```
Result:
```
Clustering graph embeddings...
1.0 +/- 0.0
```
## Real life datasets
Here you can find the real life networks we used to perform the experiments that we shown in our paper:
+ **Primary school temporal network data:**
This data set contains the temporal network of contacts between the children and teachers used in the study published in BMC Infectious Diseases 2014, 14:695. [data](http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/)
+ **Air Transportation Multiplex:** 
This is a multiplex network composed by many airlines operating in Europe. The dataset contains up to thirty-seven different layers each one corresponding to a different airline. [data](http://complex.unizar.es/~atnmultiplex/)

## References

[1] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 2010. Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. J. Mach. Learn. Res. 11 (December 2010), 3371-3408.

[2] John A. Lee, Diego H. Peluffo-Ordóñez, Michel Verleysen, Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure,Neurocomputing, Volume 169, 2015,
Pages 246-261.


