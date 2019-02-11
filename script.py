import labelClassify
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

chamadas = [("vidros.csv",[4,4,4,4,4,4,4,4,4],10)]

for i in chamadas:
	bd = pd.read_csv(i[0],sep=',',parse_dates=True)
	per_tst = 0.1
	results = pd.DataFrame(columns=['Acuracia', 'Recall', 'Precision'])
	while per_tst < 0.3:
		medias = [0]*3
		for j in range(10):
			rotular = labelClassify.LABELROTULATOR(bd, per_tst, i[1], 60, i[2], 1)
			medias[0] += rotular.metrics[0]/10
			medias[1] += rotular.metrics[1]/10
			medias[2] += rotular.metrics[2]/10
		results.loc[per_tst] = medias
		per_tst += 0.1
	print(results)
	results.to_csv('Resultado-'+str(i[0].split('.')[0])+'.csv', sep=',' )


'''

("iris.csv", [3,3,3,3], 10),("vidros.csv",[4,4,4,4,4,4,4,4,4],15),("sementes.csv", [3,3,3,3,3,3,3], 5)

'''