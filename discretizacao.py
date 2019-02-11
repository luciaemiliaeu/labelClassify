import pandas as pd
import numpy as np

class discretizador(object):
	def __init__(self, db, vector_num_faixas, metodo, attr_cluster):
		self.db = db
		self.vector_num_faixas = vector_num_faixas
		self.metodo = metodo 
		self.attr_cluster = attr_cluster
		self.data, self.ddb, self.infor = self.discretize_db()

	def discretize_db(self):
		cluster = self.db.loc[:,self.attr_cluster]
		data = self.db.drop([self.attr_cluster], axis=1)
		values = data.get_values()    
		
		ddb = []
		infor = []

		for j in range(0, data.shape[1]):
			if self.metodo is "EWD":
				disc_attb = pd.cut(values[:,j], bins = self.vector_num_faixas[j], labels = False, retbins= True)
			elif self.metodo is "EFD":
				disc_attb = pd.qcut(values[:,j], self.vector_num_faixas[j], labels = False, retbins = True, duplicates = 'drop')
			ddb.append(disc_attb[0])
			infor.append(disc_attb[1])

		ddb = np.asarray(ddb, dtype = 'int32')
		
		for x in range (0, data.shape[1]):
		   data.loc[:,data.columns[x]] = [y[x] for y in ddb.T]
		data[self.attr_cluster] = cluster
		
		return ddb,data, infor
	