# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np 
import copy 
import random
import pandas as pd
from Autoencoder import *
from multi_scale import mssne_implem
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import SpectralClustering


class Nets:


	def __init__(self, name, seed=0):
		self.seed = seed
		self.nets = self.choose_net(name)
		self.labels = self.get_nets_labels(self.nets,'type')
		self.autoencoder = Autoencoder(self.nets)
	
	
	def set_seed(self,val):
		self.seed = val	

	def get_seed(self):
		return self.seed

	def get_labels(self):
		return self.labels		


	def choose_net(self, name_net):
		if 	name_net=="ER":
			networks = self.load_ER()
		
		elif name_net=="Mixed":
			networks = self.load_Mixed()
			
		elif name_net=="LFR":
			networks = self.load_LFR()
		
		elif name_net=="dynamic":
			networks = self.load_dynamic()
		
		return networks
	


	def load_ER(self):
		num_nets = 600
		N = 81
		
		G1 = nx.fast_gnp_random_graph(N,0.05, seed=0) 
		G2 = nx.fast_gnp_random_graph(N,0.075, seed=0) 
		G3 = nx.fast_gnp_random_graph(N,0.10, seed=0) 

		nets = dict()
		for index in range(0,num_nets):
			
			self.set_seed(index)

			if index < 200:
				net = copy.deepcopy(G1) 
				G = self.nodes_permutation(net)		
				type = 0
			elif index < 400:
				net = copy.deepcopy(G2) 
				G = self.nodes_permutation(net)		
				type = 1
			elif index < 600:
				net = copy.deepcopy(G3) 
				G = self.nodes_permutation(net)			
				type = 2
						
			nets[index] = {'network': G,'type': type}

		return nets				

	def load_Mixed(self):
		num_nets = 600 # change below also, to 500
		N = 81
		m = 3
		p = 0.075
		
		nets = dict()
		
		for index in range(0,num_nets):
			
			if index < 300:

				G    =  nx.fast_gnp_random_graph(N,p, seed=index) 
				type =  0
				
			else:	
				G = nx.barabasi_albert_graph(N,m, seed=index)
				type = 1

			nets[index] = {'network': G, 'type': type}

		return nets		
	
	def load_LFR(self):

		num_nets = 1000
		
		nets = dict()
		for index, i in enumerate(range(1,num_nets + 1)): 
							
			if i <= 500:
				
				edges = pd.read_csv('data/LFR/nets01/network'+str(i)+'.dat', sep='\t', header=None)
				groups =pd.read_csv('data/LFR/nets01/community'+str(i)+'.dat', sep='\t', header=None)
				type = 0
			else:	
				j = i%500
				if j == 0:
					j = 500
				edges = pd.read_csv('data/LFR/nets05/network'+str(j)+'.dat', sep='\t', header=None)
				groups =pd.read_csv('data/LFR/nets05/community'+str(j)+'.dat', sep='\t', header=None)
				type = 1
			
			
			G = nx.from_edgelist(edges.values)
			mapping = dict(zip(groups[0].values, groups[1].values))
			
			nx.set_node_attributes(G, mapping, 'group')		
			G = nx.convert_node_labels_to_integers(G, first_label=0)
			
			G = self.nodes_permutation(G)	
			nets[index] = {'network': G, 'num_com': len(groups[1].unique()), 'type': type}

		return nets	

	def rewiring_edges(self, G,prob_rewiring, t):
		num_edges = G.number_of_edges()
		edges = np.array(G.edges())
		edges_to_rewire = int(prob_rewiring*num_edges)
		np.random.seed(t)

		#G = nx.double_edge_swap(G, nswap=edges_to_rewire, max_tries=2000)
		
		mask = np.random.choice(num_edges,num_edges, replace=False)
		edges = edges[mask]
		selected_edges = edges[0:edges_to_rewire]
		G.remove_edges_from(selected_edges)

		G_com = nx.complement(G)
		num_edges = G_com.number_of_edges()
		edges = np.array(G_com.edges())
		mask = np.random.choice(num_edges,num_edges, replace=False)
		edges = edges[mask]
		selected_edges = edges[0:edges_to_rewire]
		G.add_edges_from(selected_edges)
		
		return G

	def del_add_edges(self,G, prob_del, prob_add):
		
		A = nx.adjacency_matrix(G).toarray()

		for i in range(0,A.shape[0]):
			for j in range(i, A.shape[0]):
				if i!=j:
					if A[i,j] == 1: # prob. delete an edge
						A[i,j] = np.random.binomial(1,1 - prob_del)
						A[j,i] = A[i,j]
					else:
						A[i,j] = np.random.binomial(1,prob_add)
						A[j,i] = A[i,j]
		net = nx.from_numpy_matrix(A)
		return net				

	def load_dynamic(self):
		net_order     = 81
		prob_rewiring = 0.02
		prob_add      = 0.015
		prob_del      = 0.015
		t_max         = 600

		nets = dict()

		net = nx.fast_gnp_random_graph(net_order,0.08)
		
		np.random.seed(self.seed) 
		G = self.nodes_permutation(net)
	
		for t in range(0,t_max): 
			
			
			if t < 200:
				G = self.rewiring_edges(G, prob_rewiring,t)
				G = self.del_add_edges(G,prob_del, prob_add)
				type = 0
			
			elif t < 400:
				G = self.rewiring_edges(G, prob_rewiring,t)
				G = self.del_add_edges(G,0.2, 0.2)
				
				type = 1
			
			elif t < 600:
				G = self.rewiring_edges(G, prob_rewiring, t)
				G = self.del_add_edges(G,0.8, 0.8)
				
				type = 2

			
			nets[t] = {'network': G, 'type': type}	
			G = copy.deepcopy(nets[t]['network']) 			

		return nets	

	def nodes_permutation(self,net):
		
		np.random.seed(self.seed) 
		perm = np.random.choice(net.nodes(), net.order(), replace=False)
		nt   = dict(zip(net.nodes(),perm))
		net  = nx.relabel_nodes(net,nt)
		
		return net	

	def get_nets_labels(self, nets, name):
		
		return [nets[i][name] for i in range(0,len(nets))]



	def plot_embs(self, X_lds, labels, tit, labs=0, classes_names=0):
    
	    N= len(set(labels))			
	    cmap = plt.cm.jet
	    # extract all colors from the .jet map
	    cmaplist = [cmap(i) for i in range(cmap.N)]
	    # create the new map
	    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	    bounds = np.linspace(0,N,N+1)
	    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
			
	    a1 = plt.scatter(X_lds[:, 0], X_lds[:, 1],  c=labels, cmap=cmap, s=20, norm=norm)
	    
	    if labs==1:    
	       for i in range(X_lds.shape[0]):              
	           plt.text(X_lds[i, 0], X_lds[i, 1], str(i), fontdict={'size': 7})

	    plt.xlim([2*X_lds.min(),2*X_lds.max()])
	    plt.ylim([2*X_lds.min(),2*X_lds.max()])
	    plt.title(tit)

	    lp = lambda i: plt.plot([],color=a1.cmap(a1.norm(i)), ms=10.0, mec="none", ls="", marker="o")[0]                
	    handles = [lp(i) for i in list(set(labels))]
	    plt.legend(labels=classes_names, loc=2)

	    plt.show()	

	
	def visualize_mssne(self):

		init = 'random'
		print("Applying Multi-scale SNE")

		X_lds =	mssne_implem(X_hds=self.autoencoder.embs, init=init, n_components=2, dm_hds=None); 

		self.plot_embs(X_lds, self.labels,'Embeddings', classes_names=list(set(self.labels)))


	def visualize_tsne(self):


		print("Applying t-SNE")

		X_lds = TSNE(n_components=2).fit_transform(self.autoencoder.embs)

		self.plot_embs(X_lds, self.labels,'Embeddings', classes_names=list(set(self.labels)))	


	
	def clustering(self):
	
		n_clus = len(set(self.labels))

		print("\n")
		print("Clustering graph embeddings...")
		sclus = SpectralClustering(n_clusters=n_clus, affinity='precomputed', random_state=123)
		sclus.fit(self.autoencoder.sim_mat)
		
		nmi = metrics.normalized_mutual_info_score(self.labels, sclus.labels_)

		return nmi

		



	