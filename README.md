# Sobre o repositório
Esse repositório é uma atividade de um projeto de pibic do IFG Anápolis, onde mostra passos para utilização do GraphRec Utilizando o dataset [MovieLens 100k](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k).

## GraphRec
O GraphRec é um algoritmo de recomendação que leva em conta atributos dos usuários e dos itens, utilizando de uma técnica similar à matriz de fatoração, mas com a construção de features latentes não-lineares que podem absorver caracteristicas dos usuários ou dos itens, ou até mesmo de ambos, onde é a graph feature. Os vetores latentes podem então ser combinados para conseguir o rating de uma relação item-usuário.

O algoritmo é apresentado no artigo: [Rashed, Ahmed, Josif Grabocka, and Lars Schmidt-Thieme. "Attribute-aware non-linear co-embeddings of graph features."13th ACM Conference on Recommender Systems (RecSys). 2019.](https://www.ismll.uni-hildesheim.de/pub/pdfs/Ahmed_RecSys19.pdf)

O link do código pode ser encontrado em: https://github.com/ahmedrashed-ml/GraphRec

## Requisitos: 
    python==3.6.6
	pandas==1.1.5
	tensorflow==1.15.5
	matplotlib==3.3.4
	numpy==1.18.5
	six==1.16.0
	scikit_learn==0.24.2

## Execução
O código para execução do graphrec pode ser encontrado no arquivo [main.ipynb](https://github.com/znehAC/GraphRec-example/blob/master/main.ipynb).

O algoritmo precisa receber um dataset ([data/u.data](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k)) que contem relações de item (id), usuário (id) e a nota dada. Pode ser usado também as features externas ([data/u.user](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k) ou [data/u.item](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k)), onde é preciso então de outro dataset contendo o id do item/usuário e atributos (como as categorias de filmes).

### Hyperparâmetros
Para calibrar o algoritmo é necessário ajustar os hyperparametros, que se encontram em [graphrec.py](https://github.com/znehAC/GraphRec-example/tree/master/utils/graphrec.py).

	Varia de acordo com o dataset, valores para o movielens100k:
		USER_NUM = 943		#numero de usuarios
		ITEM_NUM = 1682 	#numero de itens
		
		Sem graph features e sem features externas
			MFSIZE=40	#parametros do modelo
			UW=0.08
			IW=0.06
			LR=0.0002
			EPOCH_MAX = 601
		Sem graph features e com features externas
			MFSIZE=40	#parametros do modelo
			UW=0.08
			IW=0.06
			LR=0.0002
			EPOCH_MAX = 311
		Com graph features
			MFSIZE=50	#parametros do modelo
			UW=0.05
			IW=0.02
			LR=0.00003
			EPOCH_MAX = 196
