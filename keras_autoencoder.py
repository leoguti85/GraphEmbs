from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import regularizers
import argparse
from param_datasets import *
from Nets import *
from Plotters import *
from sklearn import preprocessing
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import community
from networkx.readwrite import json_graph
import json
from wavelets import WaveletAnalysis
from wavelets import Ricker
from scipy.signal import butter, lfilter, freqz
from input_format import *
from graph_metrics import *
from scipy.linalg import eig
import random as rn
import os
from keras import backend as K
#-- Reproducibility
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from sklearn.cluster import AgglomerativeClustering
import kmedoids
from sklearn.cluster import SpectralClustering

'''
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(123)
rn.seed(123)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''
# run keras_autoencoder.py -f school -epochs 15 -h1_dim 800

np.random.seed(1)
pl = Plotters()

mpl.interactive(True)
np.set_printoptions(linewidth=999999)
plt.close('all')


def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
	else:
		print "Toc: start time not set"

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def save_json_net(net):
	data = json_graph.node_link_data(net)	
	with open('d3j/net.json', 'w') as outfile:
		json.dump(data, outfile)
		print("Saving json netork... ")

def get_hour_labels(nets):

	y_hours = []
	for i in nets.keys():
		time_i = nets[i]['time']
		y_hours.append(time_i.hour)

	y_hours = np.array(y_hours)
	le = preprocessing.LabelEncoder()
	
	le.fit(y_hours)
	y_train = le.transform(y_hours)
	classes_names = list(le.inverse_transform(list(set(y_train)))) 
	
	return (y_train, classes_names)

def get_activations(model, layer, X_batch):

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(layer).output,])
    
    activations = get_activations([X_batch,0])
    
    return activations

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='f', help='Name of network')
	parser.add_argument('-h1_dim', dest='h1_dim', help='hidden dimension')
	parser.add_argument('-epochs', dest='epoch', help='number of epochs')
	
	return parser.parse_args()

def distances_embeddings(dist_mat):
	euc_dist = []
	
	for i in range(0,dist_mat.shape[0]-1):
		j = i + 1	
		euc_dist.append(dist_mat[i,j])
		
	euc_dist = np.array(euc_dist)
	return euc_dist

def get_flattened_emb(data):

	embs = []
	for i in range(0,data.shape[0]):
		embs.append(data[i].flatten())

	embs = np.array(embs)
	
	return embs

def get_noise(data,p):
	res = []
	for i in range(data.shape[0]):
		res.append(np.random.binomial(1,p, data.shape[1])*-1)
	return np.array(res)	
#################################################################################
args = parse_args()

# ---- getting args parameters -----------------------
name = args.f
nb_epoch = int(args.epoch)
h1_dim = int(args.h1_dim)

net_code = param_map[name]['net_code']
batch_size = param_map[name]['bz']
lr = param_map[name]['lr']


#penalty = param_map[name]['penalty']
penalty = 0.00

#------ Load expe model ----------------------------------
i_max = 1
all_signals = []
all_raw_signals = []
dist_matrices = []

tic();
for i in range(0,i_max):

	print i
	data = Nets(net_code,i)
	nets = data.nets

	#---------------------------------------------------

	# --- Load Features vectors and Labels ------------------------------------------------------------
	#X = get_flattened_emb(adjacency_matrix(nets))
	#X = get_flattened_emb(geodesic_matrix_compute(nets))

	X = triangular_adjacency_matrix(nets,3)
	#X = triangular_geodesic_matrix(nets)
	#X = triangular_laplacian_matrix(nets, normal=True)
	#X = get_flattened_triangular_emb(adj_matrix(name)) # from mat

	# Parameters for denoising autoencoder
	nb_visible = X.shape[1]
	nb_hidden = h1_dim

	#pdb.set_trace()
	train_index = range(0,X.shape[0])
	#np.random.shuffle(train_index)

	x_train = X[train_index]

	noise_matrix = get_noise(x_train,0.1) 
	x_train_noisy = x_train + noise_matrix
	x_train_noisy[x_train_noisy<0] = 0
	#x_train_noisy[x_train_noisy>1] = 1
	

	#x_train_noisy = np.multiply(x_train,np.random.binomial(1,0.9, size = x_train.shape))

	
	print("Training shape: ", x_train.shape)
	print("Hidden dimension: ", h1_dim)
	print("Learning rate: ", lr)
	print("Batch size: ", batch_size)
	print("Penalty:  ",penalty)


	#sc = preprocessing.StandardScaler().fit(x_train) #mix
	sc = preprocessing.MinMaxScaler().fit(x_train) 
	x_train = sc.transform(x_train)
	x_train_noisy = sc.transform(x_train_noisy)


	# Build autoencoder model
	input_net = Input(shape=(nb_visible,))
	encoded = Dense(nb_hidden, activation='tanh', name='h1_layer', kernel_regularizer=regularizers.l1(penalty))(input_net)
	#encoded = BatchNormalization(axis=-1)(encoded)
	decoded = Dense(nb_visible, activation='sigmoid', name='output')(encoded)


	autoencoder = Model(input_net, decoded)
	adam = Adam(lr=lr) 
	#adam = RMSprop(lr=lr) 
	autoencoder.compile(optimizer=adam, loss='binary_crossentropy'); print("*** Cross-Entropy loss... ****");
	#autoencoder.compile(optimizer=adam, loss='mean_squared_error'); print("*** Mean Square Error loss... ****");

	# Train
	autoencoder.fit(x_train_noisy, x_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)
	               
	# Evaluate
	evaluation = autoencoder.evaluate(x_train, x_train, batch_size=batch_size, verbose=1)
	print('\nSummary: Loss over the dataset: %.2f' % evaluation)

	################ Getting activations ##############################

	layer = 'h1_layer'
	embs = get_activations(autoencoder, layer, x_train)[0]
	embs_noise = get_activations(autoencoder, layer, x_train_noisy)[0]
	#embs = embs.astype('float16')
	
	################# Visualization ####################################
	#yy = np.repeat('black',embs.shape[0])
	#yy, classes_names = y_type(nets,'type')
	#yy, classes_names = y_hours(nets)

	#pl.show_mds(embs, yy, 'mds',classes_names=0, labs=0)  
	#pl.show_tsne(embs, yy, 'tsne',classes_names=0, labs=0)  

	################# Computing simillarities ##################################
	
	# Euclidean similarity	
	dist =  metrics.pairwise.euclidean_distances(embs)
	dist_sc = (dist - dist.min())/(dist.max()-dist.min()) # scaling dist matrix
	emb_sim = 1 - dist_sc								# euclidean similarity

	# RBF similarity
	emb_sim_rbf = metrics.pairwise.rbf_kernel(embs)		# rbf kernel similarity
	
	
	# Euclidean noise embeedings	
	dist_noise =  metrics.pairwise.euclidean_distances(embs_noise)
	dist_noise_sc = (dist_noise - dist_noise.min())/(dist_noise.max()-dist_noise.min())
	emb_noise_sim = 1 - dist_noise_sc								
	

	# Raw data similarty 
	dist_raw =  metrics.pairwise.euclidean_distances(x_train)
	dist_raw_sc = (dist_raw - dist_raw.min())/(dist_raw.max() - dist_raw.min())
	raw_sim = 1 - dist_raw_sc							


	################# Distances signal ########################################

	dist_signal = distances_embeddings(dist)
	dist_noise_signal = distances_embeddings(dist_noise)
	#dist_signal = (dist_signal - dist_signal.min())/(dist_signal.max() - dist_signal.min())
	#euc_dist = (euc_dist - euc_dist.mean())/(euc_dist.std())

	# raw data signal
	dist_signal_raw = distances_embeddings(dist_raw)

	yy, classes_names = y_type(nets,'type')  
	#yy, classes_names = y_type(nets,'num_com')

	################  Saving Dataa ########################################	
	
	np.savetxt("dist_matrices/"+name+"_boe_sim.csv",dist_raw, fmt='%.8f')
	np.savetxt("dist_matrices/"+name+"_embs_sim.csv",dist, fmt='%.8f')
	np.savetxt("dist_matrices/"+name+"_embs_rbf_sim.csv",emb_sim_rbf, fmt='%.8f')
	
	np.savetxt("embs/"+name+"_embs_"+str(h1_dim)+".csv",embs, fmt='%.8f')
	
	# for muti scale
	np.savetxt("embs/y.csv",yy, fmt='%d')
	pd.DataFrame(classes_names).to_csv('embs/class_names.csv')


	########### Clustering ##########################################################
	n_clus = len(set(yy))

	print("clustering graph embeddings...")
	sclus = SpectralClustering(n_clusters=n_clus, affinity='precomputed', random_state=123)
	sclus.fit(emb_sim)
	#sclus = SpectralClustering(n_clusters=n_clus, random_state=123, gamma=0.01)
	#sclus.fit(embs)
	
	nmi = metrics.normalized_mutual_info_score(yy,sclus.labels_)
	all_signals.append(nmi)
	print "Euclidean: "+str(nmi)


	sclus.fit(raw_sim)
	nmi = metrics.normalized_mutual_info_score(yy,sclus.labels_)
	all_raw_signals.append(nmi)
	print "Raw: "+str(nmi)

	sclus.fit(emb_sim_rbf)
	nmi = metrics.normalized_mutual_info_score(yy,sclus.labels_)
	print "RBF: "+str(nmi)


	dist_matrices.append(dist)
	###################################################################################


#-----------------------------------------------------------------------------------------
all_signals = np.array(all_signals)
all_raw_signals = np.array(all_raw_signals)
dist_matrices = np.array(dist_matrices)
toc();


#findchangepts(eucsignal,'MaxNumChanges',3,'Statistic','mean')
#L = nx.normalized_laplacian_matrix(nets[0]['network']).toarray()
#w, vl, vr = eig(L, left=True)


#----- JSON -----------------------
#save_json_net(nets[100]['network'])
#G_sim = nx.from_numpy_matrix(rbf_sim)
#part=community.best_partition(G_sim, weight='weight')


'''
feat_num = 0

W=autoencoder.get_weights()[0]
df = pd.DataFrame(W)
top5 = df[feat_num].sort_values(ascending=False).head()


N = nets[0]['network'].order()

nodes = []
for edge in top5.index.values:
	nodes.append(index2xyz(edge,N))
 
top_nodes = list(set([x for sub in nodes for x in sub]))
'''

#---------------------------------------------------------------


#pl.show_tsne(embs, yy,'Graph embeddings',classes_names=classes_names,labs=0)

'''
base = 500
print np.mean(embs[0:base].flatten())
print np.mean(embs[base:2*base].flatten())
print np.mean(embs[2*base:3*base].flatten())
print np.mean(embs[3*base:4*base].flatten())
'''

'''
plt.title("ER networks - neuron activations ")

#plt.hist(embs.flatten(), alpha=0.6, label=['<d> = 4'], bins=20)
#plt.axvline(np.mean(embs.flatten()), color='k',linestyle='dashed', linewidth=1)


plt.hist(embs[0:base].mean(axis=1), alpha=0.6, label=['<d> = 4'], bins=20)
plt.axvline(np.mean(embs[0:base].flatten()), color='k',linestyle='dashed', linewidth=1)

plt.hist(embs[base:2*base].mean(axis=1), alpha=0.6, label=['<d> = 6'], bins=20)
plt.axvline(np.mean(embs[base:2*base].flatten()), color='k',linestyle='dashed', linewidth=1)


plt.hist(embs[2*base:3*base].mean(axis=1), alpha=0.6, label=['<d> = 8'], bins=20)
plt.axvline(np.mean(embs[2*base:3*base].flatten()), color='k',linestyle='dashed', linewidth=1)

plt.hist(embs[3*base:4*base].mean(axis=1), alpha=0.6, label=['<d> = 10'], bins=20)
plt.axvline(np.mean(embs[3*base:4*base].flatten()), color='k',linestyle='dashed', linewidth=1)


plt.legend()
'''

#comm = [nets[i]['num_com'] for i in range(0,len(nets))]
#pd.DataFrame(comm).to_csv('comm.csv', index=None, header=None)
