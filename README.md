# GraphEmbs

## Unsupervised Network Embeddings for Graph Visualization, Clustering and Classification


In this work we provide an unsupervised approach to learn embedding representation for a collection of graphs so that it can be used in numerous graph mining tasks. By using an unsupervised neural network approach on input graphs, we aim to capture the underlying distribution of the data in order to discriminate between different class of networks. Our method is assessed empirically on
synthetic and real life datasets and evaluated in three different tasks: graph clustering, visualization and classification. 

This code was tested on Debian GNU/Linux 8.11 (jessie), python 3.5.2

## Usage
The script run_synthetic_graphs.py will allow to reproduce the visualization and clustering results reported in the above paper.
### Example
Learning graph embeddings for Erdős–Rényi networks generated from different parameters.
Visualizatiton is done with Multi-scale SNE tool [1]:

```
python run_synthetic_graphs.py -f ER -opc 0
```
![test](https://github.com/leoguti85/GraphEmbs/blob/master/images/ER.png | width=48)

Clustering graph embeddings for Erdős–Rényi networks, evaluationg its performance with the normalized mutual information metric:

```
python run_synthetic_graphs.py -f ER -opc 1
```
```
Clustering graph embeddings...
1.0 +/- 0.0
```



