import numpy as np
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split

class classificador(object):
	def __init__(self, metodo, data, perc_trein, folds):
		self.data = data
		self.metodo = metodo
		self.perc_test = (100-perc_trein)/100
		self.folds = folds
		self.acuracia = self.classificar()	

	def classificar(self):
		acuracia = [0]*self.data.shape[1]
		for i in range(0,self.folds):
			for j in range(0, self.data.shape[1]):
				x_train, x_test, y_train, y_test = self.trainTest(self.data, j)
				y_train=np.asarray(y_train, dtype="|S6")
				y_test=np.asarray(y_test, dtype="|S6")
				clf = self.metodo
				
				if x_train.size == 0:
					acuracia[j] += 0
				else: 
					clf.fit(x_train, y_train)
					acuracia[j] += clf.score(x_test, y_test)
		
		acuracia = [i/self.folds for i in acuracia]
		resultado = list(zip(range(0,self.data.shape[1]), acuracia))
		return resultado

	def trainTest(self, data, attr):
		Y = data.loc[:,data.columns[attr]].get_values()
		X = data.drop(data.columns[attr], axis=1).get_values()
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=self.perc_test)
		return x_train, x_test, y_train, y_test

def classifica_bd(grupos, attr_cluster, porc_trein, folds):
	result = []
	for grupo in grupos:
		classif = classificador(mlp(max_iter=2000), grupo, porc_trein, folds)
		result.append(classif.acuracia)	
	return result
