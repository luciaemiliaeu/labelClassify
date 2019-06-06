

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

class MLPClas(object):
	def __init__ (self, base, perc_test):
		#Separação de conjuntos de treino e teste 
		self.X, self.Y, self.X_train, self.X_test, self.y_train, self.y_test = self.preparacao(base, perc_test)

		MLP = self.treinar(self.X_train, self.y_train)
		self.metrics, y, self.acertos, self.erros = self.classificar(MLP, self.X_test, self.y_test)
	
	def preparacao(self, data, perc_test):		
		Y = data.loc[:,'classe'].get_values()
		X = data.drop(['classe'],axis = 1).get_values()
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= perc_test)
		return X, Y, X_train, X_test, y_train, y_test

	def treinar(self, x, y):
		clf = mlp(max_iter=2000)
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

#"iris.csv","sementes.csv",
chamadas = [ "vidros.csv"]

for i in chamadas:
	bd = pd.read_csv(i,sep=',',parse_dates=True)
	per_tst = 0.1
	results = pd.DataFrame(columns=['per_tst', 'Acuracia', 'Recall', 'Precision'])
	while per_tst < 1.0:
		for j in range(10):
			print(i + " rodando com "+ str(per_tst))
			rotular = MLPClas(bd, per_tst).metrics
			rotular.insert(0,per_tst)
			results.loc[results.index.size,:] = rotular
			results.to_csv('MLP-'+str(i[0].split('.')[0])+'.csv', sep=',' )
		per_tst += 0.10
		per_tst = round(per_tst,2)



