import argparse
from Nets import *
from multi_scale import mssne_implem

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='Name of network')	
	return parser.parse_args()


args  =  parse_args()
name  =  args.f

data  =  Nets(name)
#X_hds


#X_lds = mssne_implem(X_hds=X_hds, init=init, n_components=2, dm_hds=None); print("Without similarity matrix");