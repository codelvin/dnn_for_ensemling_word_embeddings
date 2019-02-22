from __future__ import division

from abc import ABCMeta, abstractmethod
from itertools import combinations
from tqdm import tqdm
import numpy as np
#import tensorflow as tf
import os
import pandas as pd
import csv
import math
# autoencoder on big dataset
class AEBD(object):
	def generateBigSet(self):
		self.bigset = []
		cnt = 0
		for root, dirs, files in os.walk('./Test_Input/'):
			for filename in files:
				print(cnt, filename)
				word1 = []
				word2 =[]
				ID = []
				score = []
				with open(os.path.join(root, filename),'r') as f:
					for line in f.readlines():
						line = line.strip().split(',')
						word1.append(line[0])
						word2.append(line[1])
						score.append(line[2])
						ID.append(cnt)
				cnt+=1
				score = self.auto_scale(np.asarray(score, dtype = 'float32'))
				self.bigset = self.bigset+list(zip(word1, word2, ID, score))
		
		# cnt=0
		# for i in range(len(self.bigset)-1):
		# 	for j in range(i+1, len(self.bigset)):
		# 		if (self.bigset[i][0] == self.bigset[j][0] and (self.bigset[i][1] == self.bigset[j][1])) or (self.bigset[i][0] == self.bigset[j][1] and (self.bigset[i][1] == self.bigset[j][0])):
		# 			if self.bigset[i][-2] != self.bigset[j][-2]:
		# 				print(self.bigset[i])
		# 				print(self.bigset[j])
		# 				cnt+=1
		# print(cnt)
	def onehot(self, number):
		tmp = [0]*9
		tmp[number] = 1
		return np.asarray(tmp, dtype='float32')

	def auto_scale(self, array):
		factor = 1/math.ceil(np.max(array))
		return np.multiply(array, factor)

	def generateTrainingData(self, embfile1, embfile2):
		emb1 = pd.read_csv(embfile1, sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
		emb2 = pd.read_csv(embfile2, sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
		self.src_dict = {}

		for each in tqdm(self.bigset):
			if each[0] in emb1.index and each[1] in emb1.index and each[0] in emb2.index and each[1] in emb2.index:
				# data augmentation
				self.src_dict[each[0]+'%'+each[1]] = np.hstack((np.asarray(each[3], \
					dtype='float32'), \
					emb1.loc[each[0]].to_numpy(), \
					emb2.loc[each[0]].to_numpy(), \
					emb1.loc[each[1]].to_numpy(), \
					emb2.loc[each[1]].to_numpy(),self.onehot(each[2])))

		
		data = pd.DataFrame.from_dict(self.src_dict, orient='index')
		data.to_csv('more.csv', sep=' ', header=False, encoding='utf-8')
		del data


if __name__ == '__main__':
	aebd = AEBD()
	aebd.generateBigSet()
	aebd.generateTrainingData('skipgram300.txt','glove300.txt')