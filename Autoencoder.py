from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from keras import backend as K
import matplotlib as mpl
import networkx as nx
import numpy as np

class Autoencoder:

	def __init__(self,nets):
		self.nb_epoch   =  15
		self.batch_size =  128
		self.h1_dim     =  800
		self.lr         =  0.001
		self.noise      =  0.05
		self.X          =  self.triangular_adjacency_matrix(nets,3)
		self.embs       =  0
		self.sim_mat    =  0


	def similarity_matrix(self):
		dist =  metrics.pairwise.euclidean_distances(self.embs)
		dist_sc = (dist - dist.min())/(dist.max()-dist.min()) 
		emb_sim = 1 - dist_sc						
		return emb_sim		


	def triangular_adjacency_matrix(self, nets, power=1):
		
		X = []
		for g in nets.keys():
			A = nx.adjacency_matrix(nets[g]['network'])
			A = np.linalg.matrix_power(A.toarray(),power)

			indices = np.triu_indices_from(A)
			X.append(A[indices])
			#X.append(A.toarray().flatten())

		X = np.array(X)	

		return X	

	def get_noise(self,data,p):
		res = []
		for i in range(data.shape[0]):
			res.append(np.random.binomial(1,p, data.shape[1])*-1)
		return np.array(res)	
		

	def get_activations(self,model, layer, X_batch):

		get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer(layer).output,])
		
		activations = get_activations([X_batch,0])
		
		return activations

	def train(self):

		train_index = range(0,self.X.shape[0])
		nb_visible  = self.X.shape[1]
		x_train     = self.X[train_index]
		

		noise_matrix = self.get_noise(x_train,self.noise) 
		x_train_noisy = x_train + noise_matrix
		x_train_noisy[x_train_noisy<0] = 0
		
		print("Training shape: ", x_train.shape)
		print("Hidden dimension: ", self.h1_dim)
		print("Learning rate: ", self.lr)
		print("Batch size: ", self.batch_size)


		sc = preprocessing.MinMaxScaler().fit(x_train) 
		x_train = sc.transform(x_train)
		x_train_noisy = sc.transform(x_train_noisy)


		# Build autoencoder model
		input_net = Input(shape=(nb_visible,))
		encoded = Dense(self.h1_dim, activation='tanh', name='h1_layer')(input_net)
		#encoded = BatchNormalization(axis=-1)(encoded)
		decoded = Dense(nb_visible, activation='sigmoid', name='output')(encoded)


		autoencoder = Model(input_net, decoded)
		adam = Adam(lr=self.lr) 
		#autoencoder.compile(optimizer=adam, loss='binary_crossentropy'); print("*** Cross-Entropy loss... ****");
		autoencoder.compile(optimizer=adam, loss='mean_squared_error'); print("*** Mean Square Error loss... ****");

		# Train
		autoencoder.fit(x_train_noisy, x_train, epochs=self.nb_epoch, batch_size=self.batch_size, shuffle=True, verbose=1)
					   
		# Evaluate
		evaluation = autoencoder.evaluate(x_train, x_train, batch_size=self.batch_size, verbose=1)
		print('\nSummary: Loss over the dataset: %.2f' % evaluation)

		################ Getting activations ##############################

		layer = 'h1_layer'
		self.embs     =  self.get_activations(autoencoder, layer, x_train)[0]
		self.sim_mat  =  self.similarity_matrix()
		print("\n")
		