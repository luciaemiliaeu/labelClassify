from collections import Counter

class rotulador(object):
	def __init__ (self, cluster, holdout_val,V):
		self.medias = [(i, acuracia*100) for i, acuracia in holdout_val]
		self.medias.sort(key=lambda x: x[1], reverse=True)
		self.min = self.medias[0][1]-V
		self.titulos = cluster.columns.values.tolist()
		self.data = cluster.values

	def rotular_bd_discretizada(self, infor, cluster_disc):
		rotulos = []
		for i in range(0, self.data.shape[1]):
			if self.medias[i][1] >= self.min: 
				attr = self.medias[i][0]
				most_comun_value = int(Counter(cluster_disc.values[:,attr]).most_common(1)[0][0])
				rotulo = (attr,round(infor[attr][most_comun_value],2), round(infor[attr][most_comun_value+1],2))
				rotulos.append(rotulo)
		return rotulos
		
def rotular( grupos, grupos_disc, classificacao_infor, V, discretizacao_infor):
	rotulo = []
	if discretizacao_infor:		
		for i in range(0, len(grupos)):
			rotulador_ = rotulador(grupos[i], classificacao_infor[i], V)
			rotulo.append(rotulador_.rotular_bd_discretizada(discretizacao_infor, grupos_disc[i]))
	return rotulo
