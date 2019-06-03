import labelClassify
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('once')

chamadas = [("sementes.csv", "EFD", [3,3,3,3,3,3,3],5),("vidros.csv", "EWD", [4,4,4,4,4,4,4,4,4],15)]


for i in chamadas:
	bd = pd.read_csv(i[0],sep=',',parse_dates=True)
	per_tst = 0.10
	results = pd.DataFrame(columns=['per_tst','Acuracia', 'Recall', 'Precision'])
	while per_tst < 1.0 :
		for j in range(10):
			print(i[0]+" rodando com "+str(per_tst))
			rotular = labelClassify.LABELROTULATOR(bd, per_tst, i[1], i[2], 60, i[3], 10).metrics
			rotular.insert(0, per_tst)
			results.loc[results.index.size,:] = rotular
			results.to_csv('Resultado-'+str(i[0].split('.')[0])+'.csv', sep=',' )
		per_tst += 0.10
		per_tst = round(per_tst, 2)
		


'''

("iris.csv", "EFD", [3,3,3,3], 10),("vidros.csv", "EWD", [4,4,4,4,4,4,4,4,4],15),

'''