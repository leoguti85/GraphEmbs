import argparse
from Nets import *
from multi_scale import mssne_implem

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='Name of network')	
	return parser.parse_args()

# run run.py -f ER  

args  =  parse_args()
name  =  args.f


#data  =  Nets(name)
#data.autoencoder.train()    
#data.visualize_mssne()
#data.visualize_tsne()


'''
Clustering
'''

nmi_list = []
for i in range(0,10):

	data  =  Nets(name,i)
	data.autoencoder.train()   
	nmi_list.append(data.clustering())


print(str(np.mean(nmi_list))+" +/- "+str(np.std(nmi_list))) 

