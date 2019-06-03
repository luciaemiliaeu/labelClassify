import man_dados
import pandas as pd
from discretizacao import discretizador as disc
from classificacao import classifica_bd as clas
from rotulacao import rotular as rotulador 

class Rotulador(object):
	def __init__ (self, data, cluster, discre_method, bins, per_trein, V, folds):
		base = pd.DataFrame(data)
		base.loc[:,'Cluster'] = cluster
		self.bd = base
		self.frames_originais = man_dados.group_separator(base, 'Cluster')

		discretizacao = disc(base, bins, discre_method, 'Cluster')
		self.base_discretizada = discretizacao.ddb
		self.infor = discretizacao.infor
		self.frames_discretizados = man_dados.group_separator(self.base_discretizada, 'Cluster')
		
		self.classificacao = clas(self.frames_discretizados, 'Cluster', per_trein, folds)
		self.rotulo = rotulador(self.frames_originais, self.frames_discretizados, self.classificacao, V, self.infor)
