import argparse
from Nets import *
from multi_scale import mssne_implem


def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
		print("Toc: start time not set")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='Name of network')	
	parser.add_argument('-opc', dest='opc', help='Type of experiment')	
	return parser.parse_args()


# run run.py -f ER -opc 0  

args  =  parse_args()
name  =  args.f
opc  =   int(args.opc)

#-------------------------------------------------------------------------------------------------

if opc==0:

	# Generating synthetic data
	data  =  Nets(name)
	data.autoencoder.train()    


	# Visualizing graph embeddings
	print("Visualizing graph embeddings...")
	data.visualize_mssne()
	#data.visualize_tsne()

elif opc==1:

	# Clustering graph embeddings in the embedding space
	print("Clustering graph embeddings...")
	nmi_list = []
	for i in range(0,10):

		# Generate networks with different node permutations
		data  =  Nets(name,i)
		data.autoencoder.train()   
		nmi_list.append(data.clustering())


	print(str(round(np.mean(nmi_list),2))+" +/- "+str(round(np.std(nmi_list),2))) 

