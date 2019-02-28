from keras.models import Sequential
from keras.models import load_model

from keras.layers.core import Dense
from keras.layers import LSTM

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from keras.utils import plot_model
from keras.utils import np_utils
import matplotlib.pyplot as plt

from rede_recursiva import Rede_recursiva
from rede_concatenada import Rede_concatenada
from rede_convolucional import Rede_convolucional
from rede_complexa import Rede_complexa
import ferramentas_leitura as fl

import numpy as np
from numpy import array

from sklearn.model_selection import train_test_split

########################### Inicia lendo os dados, separando ou ja lendo de um arquivo
# As entradas vao ser inicialmente:
# > abertura
# > fechamento
# > minimo
# > maximo
# > media
# > volume
# As saidas vao ser inicialmente:
# > fechamento
# Deve definir quantos dias usar para treinar e descobrir o proximo
## CODIGO PARA A LEITURA DE DADOS UTILIZANDO O JOBLIB
filePath = './../DATAYEAR/COTACAO_BOVA11_2017.txt'
SampleSize = 40
Subsequences = 1
InputNumber = 6
Dias_previstos = 1
OutputPositions = np.array([1]) # variaveis a serem previstas - olhar dentro da funcao

# Organizando dados
print('Organizando dados...')
Data = fl.ReadData(filePath)
Data_norm = fl.Normalize(Data)
X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions)

X = X.reshape(X.shape[0], Subsequences, int(SampleSize/Subsequences), InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])
## Plotar aqui as saidas de fechamento e abertura em um grafico para analise visual de padroes
#plt.figure()
#plt.plot(np.arange(0, Y.shape[0]), Y[:, 0], label="Abertura real")
#plt.plot(np.arange(0, Y.shape[0]), Y[:, 1], label="Fechamento real")
#plt.title('Dados Reais')
#plt.xlabel('Dia')
#plt.ylabel('Valor normalizado')
#plt.legend()
#plt.show(block=True)

## funciona aqui a separacao entre treino, validacao e teste automatica apos o reshape mesmo
(trainx, val_testx, trainy, val_testy) = train_test_split(X        , Y        , test_size=0.3, random_state=30)
(valx  , testx    , valy  , testy    ) = train_test_split(val_testx, val_testy, test_size=0.5, random_state=30)
# Aqui separa os ultimos dias_teste dias para teste, porem o treino e validacao mantem sendo aleatorios no restante das amostras
#dias_teste = 6 # um a mais que o tanto de dias que queremos
#(testx, testy) = X[-dias_teste:-1], Y[-dias_teste:-1]
#(trainx, valx, trainy, valy) = train_test_split(X[:-dias_teste], Y[:-dias_teste], test_size=0.15, random_state=30)

########################### Aqui monta a rede usando a classe desejada
treinar = True
if treinar:
    epocas = 400 # por quantas epocas treinar

    print("Criando otimizador e rede...")
    adam = Adam(lr=0.0005)
    #rede = Rede_convolucional.montar(SampleSize, InputNumber, len(Out_posi))
    #rede = Rede_recursiva.montar(SampleSize, InputNumber, len(Out_posi))
    rede = Rede_complexa.montar(Subsequences, SampleSize, InputNumber, Dias_previstos*len(OutputPositions))
    rede.compile(optimizer=adam, loss='mse')
    # Callback para salvar melhor rede
    melhor_rede = ModelCheckpoint("Melhores_redes/atual.hdf5", save_best_only=True, verbose=1, monitor='val_loss')
    # Plotando arquitetura da rede
    plot_model(rede, "Melhores_redes/arquitetura_atual.png", show_shapes=True, show_layer_names=True)

    # Aqui acontece o treino e ajuste de pesos realmente - observar BATCH SIZE
    print("Comecando o treinamento da rede...")
    H = rede.fit(trainx, trainy, validation_data=(valx, valy), batch_size=10, epochs=epocas, callbacks=[melhor_rede], verbose=1)

########################### Testar em cima da melhor rede possivel salva anteriormente
rede2 = load_model('Melhores_redes/atual.hdf5')
saida_teste = rede2.predict(testx, verbose=1)
#saida_teste = np.zeros([Dias_previstos, len(Out_posi)])
#saida_teste = saida_teste.astype('float32')

#amostra_teste = 3 # Qual dia do conjunto de teste usar como ponto de partida para prever Dias_previstos futuros

#teste_atual = testx[amostra_teste]
#teste_complexa = testx_complexa[amostra_teste] # Rede complexa - backup necessario
#saida_real  = testy[np.arange(amostra_teste, amostra_teste+Dias_previstos)]
## Rodando aqui iterativamente para prever Dias_previstos a frente
#for d in np.arange(0, Dias_previstos):
#    teste_atual_resh = teste_atual.reshape(1, Subsequences, int(SampleSize/Subsequences), InputNumber) # para entrada na TimeDistributed, 1 sample, subsequences, timesteps/subsequences, 6 features
#    #teste_atual_resh = teste_atual.reshape(1, teste_atual.shape[0], teste_atual.shape[1]) # para entrada na LSTM, 1 sample, 50 timesteps, 6 features
#    saida_temp = rede2.predict(teste_atual_resh)
#    saida_teste[d] = saida_temp
    
#    # Rede complexa aqui
#    teste_complexa = np.delete(teste_complexa, 0, axis=0)
#    teste_complexa = np.append(teste_complexa, saida_temp, axis=0)
#    teste_atual    = teste_complexa # Pega do backup anterior modificado, entre no reshape na proxima iteracao correto

    ## Renova o vetor de entradas com o ultimo dia previsto, removendo o primeiro
    #teste_atual = np.delete(teste_atual, 0, axis=0) # remove a primeira linha de features, primeiro dia
    #teste_atual = np.append(teste_atual, saida_temp, axis=0)

########################### Plot grafico da evolucao da rede
if treinar:
    plt.style.use("ggplot")
    plt.plot(np.arange(0, epocas), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epocas), H.history["val_loss"], label="val_loss")
    plt.title('Evolution')
    plt.xlabel('Epoch #')
    plt.ylabel('Value')
    plt.legend()

# Plot para comparacao com o conjunto de teste - Precos de saida de abertura e fechamento
testy = testy.flatten()
saida_teste = saida_teste.flatten()

plt.figure()
plt.plot(np.arange(0, testy.shape[0] ), testy , label="Fechamento Real")
plt.plot(np.arange(0, saida_teste.shape[0]), saida_teste, label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor_norm')
plt.legend()
plt.show(block=True)
