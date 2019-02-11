

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import rotuladorLopes as rotulador
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE

class LABELROTULATOR(object):
	def __init__ (self, base, perc_test, discre_bins, perc_trei_rot, V, folds_rot):
		#Separação de conjuntos de treino e teste 
		self.X, self.Y, self.X_train, self.X_test, self.y_train, self.y_test = self.preparacao(base, perc_test)
		
		#Rotulaçaõ do conjunto de treino e seleção dos elementos pertencentes ao rótulo 
		#dados -> componente x dos elementos pertencentes ao rótulo
		#cluster -> componente y dos elementos pertencentes ao rótulo
		#notDados -> elementos não pertencentes ao rótulo
		self.rotulos = rotulador.Rotulador(self.X_train, self.y_train, 'EWD', discre_bins, perc_trei_rot, V, folds_rot).rotulo
		self.dados, self.cluster, self.notDados = self.selectData()
		
		#Treinamento da MPL com o dados selecionados
		#metrics -> [acuracia, recall, sensibilidade]
		#y -> saída predita pelo classificados para o conjunto de teste 
		#ronX, rongY -> elementos pretidos incorretamente
		MLP = self.treinar(self.dados, self.cluster)
		self.metrics, y, self.acertos, self.erros = self.classificar(MLP, self.X_test, self.y_test)

		#self.plotar()
	
	def preparacao(self, data, perc_test):		
		Y = data.loc[:,'classe'].get_values()
		X = data.drop(['classe'],axis = 1).get_values()
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= perc_test)
		return X, Y, X_train, X_test, y_train, y_test

	def selectData(self):
		newData=[]
		notData=[]

		data = pd.DataFrame(self.X_train)
		data.loc[:,'Cluster'] = pd.Series(self.y_train, index=data.index)
		grouped = data.groupby(['Cluster'])

		n=0
		for n_cluster, values in grouped:
			for i in values.get_values().tolist():
				if self.check(i, self.rotulos[n]):
					newData.append(i)
				else:
					notData.append(i)
			n=+1

		cluster = [i[-1] for i in newData]
		for i in newData: i.pop(-1)
		return np.asarray(newData), np.asarray(cluster), np.asarray(notData)
	
	def check(self, dado, rotulos): 	 	
		for j in rotulos:
			for i in range(len(dado)):
				if all([i == j[0], dado[i] >= j[1], dado[i] <= j[2]]):
					return True
		return False

	def treinar(self, x, y):
		clf = mlp()
		clf.fit(x, y)
		return clf
	
	def classificar(self, clf, x, y_true):
		y = clf.predict(x)
		acuracia = accuracy_score(y_true, y)
		recall = recall_score(y_true, y, average='micro')
		precision = precision_score(y_true, y, average='weighted')
		
		acertos=[]
		erros=[]
		for i in range(y.shape[0]):
			element = np.append(x[i], y[i])
			if y[i] == y_true[i]:
				acertos.append(element) 
			else:
				erros.append(element)
		return [acuracia, precision, recall], y, np.asarray(acertos), np.asarray(erros)

	def plotar(self):
		fig, axes = plt.subplots(nrows=2, ncols=2)

		#classe originais
		dimensionalX = PCA(n_components=2).fit_transform(self.X)
		axes[0,0].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.Y, cmap='Accent')
		
		#conjunto de treino
		dimensionalX = PCA(n_components=2).fit_transform(self.X_train)
		axes[0,1].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.y_train, cmap='Accent')
		
		#selecionados pelo rotulador
		dimensionalX = PCA(n_components=2).fit_transform(self.dados)
		axes[1,0].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.cluster, cmap='Accent')
		if self.notDados.shape[0]>0:
			dimensionalX = PCA(n_components=2).fit_transform(self.notDados[:,:-1])
			axes[1,0].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.notDados[:,-1], cmap='Set3', marker='^')
		
		#classificados
		dimensionalX = PCA(n_components=2).fit_transform(self.X_train)
		axes[1,1].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.y_train, cmap='Accent')
		dimensionalX = PCA(n_components=2).fit_transform(self.acertos[:,:-1])
		axes[1,1].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.acertos[:,-1], cmap='Accent')
		if self.erros.shape[0]>0:
			dimensionalX = PCA(n_components=2).fit_transform(self.erros[:,:-1])
			axes[1,1].scatter(x=dimensionalX[:,0], y=dimensionalX[:,1],c=self.erros[:,-1], cmap='Accent', marker='>')
		plt.show()