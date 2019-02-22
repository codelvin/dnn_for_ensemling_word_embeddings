import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.preprocessing as skpre
import csv
import scipy.stats
import sys
from time import gmtime, strftime

class DNN(object):
	def __init__(self):
		self.sess = tf.Session()
		self.n_splits = 10
		self.batch_size = 16
		self.epochs = 300
		self.alpha = 1e-3  # learning rate
		self.decay = 0.995
		self.weight_decay = 0.001
		self.dimensions = 1209
		self.n_hidden1 = 512
		self.n_hidden2 = 256
		self.n_hidden3 = 64
		self.ckpt = None
		self.ckpt_path = './checkpoints/'
		self.cv = True

	def get_weight(self, shape):
		xavier = tf.contrib.layers.xavier_initializer()
		var = tf.Variable(xavier(shape), dtype=tf.float32)
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.weight_decay)(var))
		return var

	def load_data(self, filename):
		data = pd.read_csv(filename, sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
		data = data.sample(frac=1)
		self.Y = data.values[:, 0]
		self.X = skpre.normalize(data.values[:, 1:], axis = 1)
		self.Word = data.index
		
		# placeholder
		self.x_placeholder = tf.placeholder(tf.float32, shape=(None, self.dimensions))
		self.y_placeholder = tf.placeholder(tf.float32, shape=(None,))
		self.lr = tf.placeholder(tf.float32, shape=[])
		self.tf_is_training = tf.placeholder(tf.bool,None)
		print("load completed!")

	def build_model(self):
		# network architecture
		weights = {'w1': self.get_weight([self.dimensions, self.n_hidden1]),\
		   'w2': self.get_weight([self.n_hidden1, self.n_hidden2]),\
		   'w3': self.get_weight([self.n_hidden2, self.n_hidden3]),\
			   #'w4': tf.Variable(xavier([self.n_hidden3, self.n_hidden4])),\
		   'out': self.get_weight([self.n_hidden3, 1])}

		biases = {'b1': tf.Variable(tf.random_normal([self.n_hidden1], stddev=0.01)),
		  	'b2': tf.Variable(tf.random_normal([self.n_hidden2], stddev=0.01)),
		  	'b3': tf.Variable(tf.random_normal([self.n_hidden3], stddev=0.01)),
			  #'b4': tf.Variable(tf.random_normal([self.n_hidden4], stddev=0.01)),
		  	'out': tf.Variable(tf.random_normal([1], stddev=0.01))}
	
		self.l1 = tf.nn.relu(tf.matmul(self.x_placeholder, weights['w1']) + biases['b1'])
		self.dropout1 = tf.layers.dropout(self.l1, rate=0.5, training = self.tf_is_training)  # when it is training, dropout works, o.w. open all gates
		self.l2 = tf.nn.relu(tf.matmul(self.dropout1, weights['w2'])+biases['b2'])
		self.dropout2 = tf.layers.dropout(self.l2, rate=0.5, training = self.tf_is_training)
		self.l3 = tf.nn.relu(tf.matmul(self.dropout2, weights['w3'])+biases['b3'])
		#self.dropout3 = tf.nn.dropout(self.l3, keep_prob=0.5)
		#self.l4 = tf.nn.relu(tf.matmul(self.dropout3, weights['w4'])+biases['b4'])
		self.predict = tf.nn.relu(tf.matmul(self.l3, weights['out'])+biases['out'])
		return self.predict

	def augment(self, X_train, Y_train):
		#permutation = list(range(300, 600)) + list(range(0,300)) + list(range(1200, 1209)) 
		permutation = list(range(600,900)) + list(range(900, 1200)) + list(range(0, 300)) + list(range(300, 600)) + list(range(1200, 1209))
		#permutation2 = list(range(600,900)) + list(range(900, 1200)) + list(range(0, 300)) + list(range(300, 600)) + list(range(1200, 1209))
		_X_train = X_train[:, permutation]
		X_train = np.vstack((X_train, _X_train))
		Y_train = np.hstack((Y_train,Y_train))
		return X_train, Y_train

	def train_model(self, X_train, Y_train, Word_train, X_test, Y_test, Word_test):
		# define some training hyper-param
		step = tf.Variable(0, trainable=False)
		rate = tf.train.exponential_decay(self.alpha, step, int(20000/self.batch_size), self.decay)
		predict = self.build_model()

		# loss defined as the mse + l2 reg
		mse_loss = tf.losses.mean_squared_error(tf.reshape(self.y_placeholder, [-1, 1]), predict)
		loss = tf.losses.mean_squared_error(tf.reshape(self.y_placeholder, [-1, 1]), predict) + tf.add_n(tf.get_collection('losses'))

		optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
		train_op = optimizer.minimize(loss)
		self.ckpt = tf.train.Saver(tf.global_variables())

		self.sess.run(tf.global_variables_initializer())

		epoch_loss = []
		patience = 20
		patience_cnt = 0
		min_delta = -0.05
		# learning rate half when necessary
		min_lr_delta= -0.3
		fold_alpha = self.alpha
		alpha_patience = 5
		alpha_patience_cnt = 0

		# data augmentation on train
		X_train, Y_train = self.augment(X_train, Y_train)

		size = Y_train.shape[0] // self.batch_size
		best = float('inf')

		# loop for N epochs
		for itr in range(self.epochs):
			train_loss = 0
			val_loss = 0
			best = float('inf')
			for idx in range(size):
				batch_idx = X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
				X_batch = X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
				Y_batch = Y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
				self.sess.run(train_op, \
								feed_dict={self.x_placeholder: X_batch, self.y_placeholder: Y_batch\
									, self.lr:fold_alpha , self.tf_is_training:True})
			
				batch_loss = self.sess.run(loss, feed_dict={self.x_placeholder: X_batch, \
												self.y_placeholder: Y_batch, self.lr:fold_alpha, self.tf_is_training: False})
				train_loss += batch_loss

			# calculate val_loss and train_loss, store best model so far to checkpoints
			if self.cv == True:
				val_loss = self.sess.run(loss, feed_dict={self.x_placeholder: X_test, \
											self.y_placeholder: Y_test, self.lr:fold_alpha, self.tf_is_training:False})
				val_mse =  self.sess.run(mse_loss, feed_dict={self.x_placeholder: X_test, \
											self.y_placeholder: Y_test, self.lr:fold_alpha, self.tf_is_training:False})
				train_mse =  self.sess.run(mse_loss, feed_dict={self.x_placeholder: X_train, \
											self.y_placeholder: Y_train, self.lr:fold_alpha, self.tf_is_training:False})
				print("{} epoch {}, val_loss {}, train_loss {}, val_mse {}, train_mse {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), itr, val_loss, train_loss/size, val_mse, train_mse))
				epoch_loss.append(val_loss)
			else:
				epoch_loss.append(train_loss / size)
				print("epoch {}, loss {}".format(itr, epoch_loss))
				if epoch_loss <= best:
					self.ckpt.save(self.sess, self.ckpt_path)
			
			#early stopping
			if itr>0 and epoch_loss[itr-1]-epoch_loss[itr] > min_delta:
				patience_cnt = 0
			else:
				patience_cnt += 1
			
			if patience_cnt == patience:
				print("Early stopping")
				break
			
			# half learning rate when necessary
			if  itr>0 and epoch_loss[itr-1] - epoch_loss[itr] > min_lr_delta:
				alpha_patience_cnt = 0
			else:
				alpha_patience_cnt +=1
			
			if alpha_patience_cnt == alpha_patience or (itr>0 and itr%60==0):
				print("half learning rate....")
				fold_alpha /= 4

		# after we have trained the model, we generated prediction based on whether it is CV.
		# if it is cv, then generated the fold we ain't train. 
		# else just generated the score on training set
		if self.cv == False:
			datasetID = list(np.argmax(X_train[:, -9:], axis = 1))
			word1 = [each.split('%')[0] for each in Word_train]
			word2 = [each.split('%')[1] for each in Word_train]
			truth = list(Y_train)
			score = list(self.sess.run(self.predict, \
					feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train, self.lr:self.alpha, self.tf_is_training:False}))
		else:
			datasetID = list(np.argmax(X_test[:, -9:], axis = 1))
			word1 = [each.split('%')[0] for each in Word_test]
			word2 = [each.split('%')[1] for each in Word_test]
			truth = list(Y_test)
			score = list(self.sess.run(self.predict, \
				feed_dict={self.x_placeholder: X_test, self.y_placeholder: Y_test, self.lr:self.alpha, self.tf_is_training:False}))
		
		# results has all wordpair-ID-score-truth in dataset.
		for i in range(len(score)):
			self.results = self.results.append({'word1' : word1[i], 'word2' : word2[i], 'datasetID' : datasetID[i],\
								 'score' : score[i],'truth' : truth[i]}, ignore_index=True)
		
		# if not cross-validation, then just feed the result to evaluation
		if self.cv == False:
			self.evaluation()
			sys.exit()
		
		# delete losses in collection, o.w. they will still exist in next fold
		# can also use: params = tf.get_collection_ref(tf.GraphKeys.REGULARIZATION_LOSSES); del params[:]
		graph = tf.get_default_graph()
		graph.clear_collection("losses")

	def cross_validation(self):
		self.results = pd.DataFrame(columns = ['word1','word2','datasetID', 'score','truth'])
		cnt = 0
		for train_index, test_index in KFold(self.n_splits).split(self.X):
			X_train, X_test = self.X[train_index], self.X[test_index]
			Y_train, Y_test = self.Y[train_index], self.Y[test_index]
			Word_train, Word_test = self.Word[train_index], self.Word[test_index]

			self.train_model(X_train, Y_train, Word_train, X_test, Y_test, Word_test)
			cnt += 1
			print("{} fold of {} splits is completed".format(cnt, self.n_splits))
		self.sess.close()

	def evaluation(self):
		# correlation in different dataset.
		grouped = self.results.groupby('datasetID')
		for name, group in grouped:
			pred_cosine = group['score'].to_numpy()
			true_cosine = group['truth'].to_numpy()
			corr = scipy.stats.spearmanr(pred_cosine, true_cosine).correlation
			print("datasetID: {}, correlation: {}".format(name, corr))


if __name__ == '__main__':
	dnn = DNN()
	dnn.load_data('more.csv')
	dnn.cross_validation()
	dnn.evaluation()
	print("finished")



