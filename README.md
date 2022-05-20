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

O dataset u.user, no exemplo do movielens100k contem as colunas [id_usuario, idade, genero, ocupacao, zipcode]. Exemplo:

	[1, 24, M, technician, 85711]
	[2, 53, F, other, 94043]
	[49, 23, F, student, 76111]

O u.item possui as colunas [id_item, titulo_filme, data_lançamento, data_lançamento_video, url_IMDB, categoria1, categoria2, ..., categoria19], sendo o id do item e as colunas onde tem os valores 1 ou 0, indicando se possui ou nao a categoria. Exemplo:

	[1, 'Toy Story (1995)', 01-Jan-1995, null, http://us.imdb.com/M/title-exact?Toy%20Story%20(1995), 0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
	[254, 'Batman & Robin (1997)', 20-Jun-1997, null, http://us.imdb.com/M/title-exact?Batman+%26+Robin+(1997), 0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]

Para utilização de outro dataset, é preciso implementar um método que carregue seus dados. Exemplo:

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

E modificar o código para chamar o método e adicionar os atributos. Exemplo:

	if(UserData):
	    if(Dataset=='100k'):
	      	UsrDat=get_UserData100k()
	    UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 
	if(ItemData):
	    if(Dataset=='100k'):
			ItmDat=get_ItemData100k()
	    ItemFeatures=np.concatenate((ItemFeatures,ItmDat), axis=1) 

	linha 158



### Codigo
Para treinar o modelo carregamos o dataset que contem os id's de usuario e de item e o rating de cada relacao,
Para utilizar as features de usuario e de item o loading delas é feita no proprio código do graphrec.
Essa funcao carrega os dados e separa em 90% para treino e 10% de teste.

	df_train, df_test = get_data100k() 
	# Retorna o dataset data.u dividido em treino e teste

	def get_data100k():
		df = read_process("data/ml100k/u.data", sep="\t")
		rows = len(df)
		df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
		split_index = int(rows * 0.9)
		df_train = df[0:split_index]
		df_test = df[split_index:].reset_index(drop=True)
		return df_train, df_test

Treinando o modelo podemos selecionar se vai ser usado dados do usuario(UserData) e de item(ItemData), e se vai utilizar o graph-based features (Graph), dependendo da escolha utilizada os hyper-parametros devem ser atualizados.

	model = GraphRec(
		df_train, 
		df_test, 
		ItemData= False, 
		UserData = False, 
		Graph=False, 
		Dataset='100k') 
	
	# Retorna o modelo já treinado

Para avaliação do modelo foi separado os ids dos usuarios como id de consultas e a nota como a label.

	df_test = df_test.sort_values('user') 
	qids = df_test['user'] 					
	y_test = df_test['rate'] 				
	predictions = np.array(model.predict(df_test)).flatten()
	

Após conseguir as predições, só passar para uma métrica como ndcg ou rmse.

	ndcgs = queries_ndcg(y_test, predictions, qids) 
	print("mean ndcg:", ndcgs.mean())
	rmse = mean_squared_error(y_test, predictions, squared = False)
	print("rmse:", rmse)

## Hyperparâmetros
Para calibrar o algoritmo é necessário ajustar os hyperparametros, que se encontram em [graphrec.py](https://github.com/znehAC/GraphRec-example/tree/master/utils/graphrec.py).

O artigo apresenta dois métodos, "Com Graph-Based Features" e  "Sem Graph-Based Features", e a utilização ou não de features externas. Isso implica na mudança de parâmetros.

Também de acordo com cada dataset, é preciso alterar os seguintes parâmetros:

	USER_NUM = 943		#numero de usuarios
	ITEM_NUM = 1682 	#numero de itens

	MFSIZE=40	
	UW=0.08
	IW=0.06
	LR=0.0002
	EPOCH_MAX = 601	
	
	linha 551

sem graph features com external features 

      model = GraphRec(df_train, df_test, ItemData= True, UserData = True, Graph=False, Dataset='100k')

      mean ndcg: 0.8944496166257601
      rmse: 0.8958193504345224

com graph features sem external features 

      model = GraphRec(df_train, df_test, ItemData= False, UserData = False, Graph=True, Dataset='100k')

      mean ndcg: 0.8977134940003574
      rmse: 0.8990311527796113
