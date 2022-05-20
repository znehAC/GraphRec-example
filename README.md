# Sobre o repositório
Esse repositório atende a uma das atividades do Projeto de PIBIT/IFG intitulado: "Aplicação de Modelos Supervisionados para Sistemas de Recomendação em Instituições de Pesquisa Científica." O projeto faz parte do Grupo de Pesquisa GECOMP, do Bacharelado em Ciência da Computação do IFG e é orientado pelo professor Dr. Daniel Xavier de Sousa.
O objetivo desta atividade é descrever o passo-a-passo de como utilizar o método GraphRec, considerado atualmente o estado da arte em Sistemas de Recomendação para abordagens híbridas. Para melhor descrição, utilizamos a base de dados [MovieLens 100k](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k).

## GraphRec
O GraphRec é um método de Sistemas de Recomendação com Filtragem Colaborativa que leva em conta os atributos dos usuários e dos itens, utilizando de uma técnica similar à matriz de fatoração, mas com a aplicação de Redes Neurais e a construção de atributos latentes não-lineares que podem absorver características dos usuários ou dos itens, ou até mesmo de ambos. Os vetores latentes podem então ser combinados para conseguir a predição das avaliações e uma relação item-usuário.

O método é apresentado em: [Rashed, Ahmed, Josif Grabocka, and Lars Schmidt-Thieme. "Attribute-aware non-linear co-embeddings of graph features."13th ACM Conference on Recommender Systems (RecSys). 2019.](https://www.ismll.uni-hildesheim.de/pub/pdfs/Ahmed_RecSys19.pdf)

O link original do código pode ser encontrado em: https://github.com/ahmedrashed-ml/GraphRec

## Requisitos: 
    python==3.6.6
	pandas==1.1.5
	tensorflow==1.15.5
	matplotlib==3.3.4
	numpy==1.18.5
	six==1.16.0
	scikit_learn==0.24.2

## Execução

O método precisa receber um dataset ([data/u.data](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k)) que contêm relações de item (id), usuário (id) e a relevância atribuída do item para o usuário. Considerando a estratégia do artigo de inclusão de atributos de usuários e itens para obtenção dos vetores latentes, faz-se necessário carregar os respectivos dados [data/u.user](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k) e [data/u.item](https://github.com/znehAC/GraphRec-example/tree/master/data/ml100k).

Para utilização de outro dataset, é preciso modificar no código a leitura das features externas. Exemplo:

	if(UserData):
      if(Dataset=='1m'):
        UsrDat=get_UserData1M()
      if(Dataset=='100k'):
        UsrDat=get_UserData100k()
      UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 

    if(ItemData):
      if(Dataset=='1m'):
        ItmDat=get_ItemData1M()
      if(Dataset=='100k'):
        ItmDat=get_ItemData100k()

	linha 158
e implementar um metodo que carregue seus dados. Exemplo:

	def get_UserData100k():
		col_names = ["user", "age", "gender", "occupation","PostCode"]
		df = pd.read_csv('data/ml100k/u.user', sep='|', header=None,
		names=col_names, engine='python')
		del df["PostCode"]
		df["user"]-=1
		df=pd.get_dummies(df,columns=[ "age", "gender", "occupation"])
		del df["user"]
		return df.values
	
	linha 459

## Hyperparâmetros
Para calibrar o algoritmo é necessário ajustar os hyperparametros, que se encontram em [graphrec.py](https://github.com/znehAC/GraphRec-example/tree/master/utils/graphrec.py).

O artigo apresenta dois métodos, "Com Graph-Based Features" e  "Sem Graph-Based Features", e a utilização ou não de features externas. Isso implica na mudança de parâmetros.
Parametros para o movielens 100k (encontrados a partir da linha 551):

Sem graph features e sem features externas

	MFSIZE=40	
	UW=0.08
	IW=0.06
	LR=0.0002
	EPOCH_MAX = 601	

Sem graph features e com features externas

	MFSIZE=40	
	UW=0.08
	IW=0.06
	LR=0.0002
	EPOCH_MAX = 311	

Com graph features

	MFSIZE=50
	UW=0.05
	IW=0.02
	LR=0.00003
	EPOCH_MAX = 196

Também de acordo com cada dataset, é preciso alterar os seguintes parâmetros:

	USER_NUM = 943		#numero de usuarios
	ITEM_NUM = 1682 	#numero de itens
