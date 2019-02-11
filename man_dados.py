import pandas as pd 
import numpy as np

# Classe ReadData
# Recebe o caminho do arquivo csv já agrupado
# Retorna a base de dados discretizada, as informações da discretização e um frame de cada grupo
def read_csv(path):
	bd = pd.read_csv(path,sep=',',parse_dates=True)
	return bd
# separador_grupos - Método para divisão da base de dados em um conjunto de dataframes de acordo com o atributo cluster
# deve ser chamado apenas se a base de dados já estiver agrupada/contem o atributo "grupo"
# Recebe um DataFrame (data). Retorna um conjunto de DataFrames (frames)
def group_separator(data, attr_name):
	grouped = data.groupby(attr_name)                
	
	# cria um conjunto de dataframes, cada dataframe representa um grupo
	frames = [] 								           
	for nome, grupo in grouped:
		cluster = grupo
		cluster = cluster.drop([attr_name],axis = 1)
		frames.append(cluster)		

	return frames				
def num_instancias(data, attr_cluster):
	data = data.drop([attr_cluster], axis=1)
	num_instancias = []
	for i in range(0, data.shape[1]):
		values = data.loc[:,data.columns[i]].get_values()
		num_instancias.append(len(sorted(set(values))))
	return num_instancias


