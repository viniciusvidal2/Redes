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
#filePath = ['./../DATAYEAR/COTACAO_CMIG4_2011.txt', './../DATAYEAR/COTACAO_CMIG4_2012.txt', './../DATAYEAR/COTACAO_CMIG4_2013.txt', './../DATAYEAR/COTACAO_CMIG4_2014.txt',
#            './../DATAYEAR/COTACAO_CMIG4_2015.txt', './../DATAYEAR/COTACAO_CMIG4_2016.txt', './../DATAYEAR/COTACAO_CMIG4_2017.txt', './../DATAYEAR/COTACAO_CMIG4_2018.txt']
#filePath = ['./../DATAYEAR/COTACAO_ITUB4_2011.txt', './../DATAYEAR/COTACAO_ITUB4_2012.txt', './../DATAYEAR/COTACAO_ITUB4_2013.txt', './../DATAYEAR/COTACAO_ITUB4_2014.txt',
#            './../DATAYEAR/COTACAO_ITUB4_2015.txt', './../DATAYEAR/COTACAO_ITUB4_2016.txt', './../DATAYEAR/COTACAO_ITUB4_2017.txt', './../DATAYEAR/COTACAO_ITUB4_2018.txt']
filePath = ['./../DATAYEAR/COTACAO_PETR4_2011.txt', './../DATAYEAR/COTACAO_PETR4_2012.txt', './../DATAYEAR/COTACAO_PETR4_2013.txt', './../DATAYEAR/COTACAO_PETR4_2014.txt',
            './../DATAYEAR/COTACAO_PETR4_2015.txt', './../DATAYEAR/COTACAO_PETR4_2016.txt', './../DATAYEAR/COTACAO_PETR4_2017.txt', './../DATAYEAR/COTACAO_PETR4_2018.txt']
SampleSize = 50
Subsequences = 2
mascara_entradas = [0, 1, 0, 0, 1, 1]
InputNumber = sum(mascara_entradas)
Dias_previstos = 4
OutputPositions = np.array([1]) # variaveis a serem previstas - olhar dentro da funcao

# Organizando dados
print('Organizando dados...')
Data = fl.ReadData(filePath)
Data_norm, maximos, minimos, amplitudes = fl.Normalize(Data)

X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions, mascara_entradas)

X = X.reshape(X.shape[0], Subsequences, int(SampleSize/Subsequences), InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])
## Plotar aqui as saidas de fechamento e abertura em um grafico para analise visual de padroes
plt.figure()
plt.plot(np.arange(0, Y.shape[0]), Y[:, 0], label="Fechamento real")
plt.title('Dados Reais')
plt.xlabel('Dia')
plt.ylabel('Valor normalizado')
plt.legend()
plt.show(block=True)

### funciona aqui a separacao entre treino, validacao e teste automatica apos o reshape mesmo
#(trainx, val_testx, trainy, val_testy) = train_test_split(X        , Y        , test_size=0.3, random_state=30)
#(valx  , testx    , valy  , testy    ) = train_test_split(val_testx, val_testy, test_size=0.5, random_state=30)

########################### Aqui separa os ultimos dias_teste dias para teste, porem o treino e validacao mantem sendo aleatorios no restante das amostras
intervalo_teste = np.arange(400, 600) # epoca ainda desconhecida, amostras sequenciais
X_teste, Y_teste = X[intervalo_teste].copy(), Y[intervalo_teste].copy()

dias_teste = np.arange(0, X_teste.shape[0], Dias_previstos) # testar sobre as amostras de teste em um intervalo de Dias_previstos
(testx, testy) = X_teste[dias_teste], Y_teste[dias_teste] # Dias que realmente havera teste sobre, entrarao na funcao predict

Xtreino = np.delete(X, intervalo_teste, axis=0)
Ytreino = np.delete(Y, intervalo_teste, axis=0)
#Xtreino = np.delete(X, intervalo_teste+X.shape[0], axis=0)
#Ytreino = np.delete(Y, intervalo_teste+X.shape[0], axis=0)
(trainx, valx, trainy, valy) = train_test_split(Xtreino, Ytreino, test_size=0.25, random_state=20)

########################### Aqui monta a rede usando a classe desejada
treinar = False
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
    H = rede.fit(trainx, trainy, validation_data=(valx, valy), batch_size=100, epochs=epocas, callbacks=[melhor_rede], verbose=1)

########################### Testar em cima da melhor rede possivel salva anteriormente
rede2 = load_model('Melhores_redes/atual.hdf5')
#amostra_teste = 3 # Qual dia do conjunto de teste usar como ponto de partida para prever Dias_previstos futuros
saida_teste = rede2.predict(testx, verbose=1)

########################### Plot grafico da evolucao da rede
if treinar:
    plt.style.use("ggplot")
    plt.plot(np.arange(0, epocas), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epocas), H.history["val_loss"], label="val_loss")
    plt.title('Evolution')
    plt.xlabel('Epoch #')
    plt.ylabel('Value')
    plt.legend()

# Preparando dados para plotar
conjunto_par   = np.arange(0, testy.shape[0], 2)
conjunto_impar = np.arange(1, testy.shape[0], 2)

testey_par   = testy.copy()
testey_impar = testy.copy()
saida_teste_par   = saida_teste.copy()
saida_teste_impar = saida_teste.copy()

testey_par[conjunto_par]          = np.zeros([1, Dias_previstos*len(OutputPositions)])
testey_impar[conjunto_impar]      = np.zeros([1, Dias_previstos*len(OutputPositions)])
saida_teste_par[conjunto_par]     = np.zeros([1, Dias_previstos*len(OutputPositions)])
saida_teste_impar[conjunto_impar] = np.zeros([1, Dias_previstos*len(OutputPositions)])

#testy = testy[amostra_teste].flatten()
testey_par        = testey_par.flatten()
testey_impar      = testey_impar.flatten()
saida_teste_par   = saida_teste_par.flatten()
saida_teste_impar = saida_teste_impar.flatten()

# Plot para comparacao com o conjunto de teste - Precos de saida de abertura e fechamento
plt.figure()
plt.plot(np.arange(0, testey_par.shape[0] )    , testey_par     , 'b+', label="Fechamento Real")
plt.plot(np.arange(0, testey_impar.shape[0] )  , testey_impar   , 'k+', label="Fechamento Real")
plt.plot(np.arange(0, saida_teste_par.shape[0])  , saida_teste_par  , 'ro', label="Fechamento Calculado")
plt.plot(np.arange(0, saida_teste_impar.shape[0]), saida_teste_impar, 'mo', label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor_norm')
plt.legend()
plt.grid()

########################### Trazer de volta para valores reais
testy = testy.flatten()
saida_teste = saida_teste.flatten()
Y_real           = fl.ReturnRealValue(testy      , minimos, maximos, amplitudes, OutputPositions)
saida_teste_real = fl.ReturnRealValue(saida_teste, minimos, maximos, amplitudes, OutputPositions)

plt.figure()
plt.plot(np.arange(0, Y_real.shape[0] )         , Y_real          , label="Fechamento Real")
plt.plot(np.arange(0, saida_teste_real.shape[0]), saida_teste_real, label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor R$')
plt.legend()
plt.grid()

# Calculo do erro entre os valores reais R$
erros = Y_real - saida_teste_real
media_erros = np.mean(erros)
media_erros_abs = np.mean(np.abs(erros))
erro_max = np.max(erros)
erro_min = np.min(erros)

print('\nA media dos erros e: {}\t Sobre valores absolutos: {}'.format(media_erros, media_erros_abs))
print('\nErro minimo R$     : {}\t Erro maximo R$         : {}'.format(erro_min   , erro_max       ))

plt.figure()
plt.plot(np.arange(0, erros.shape[0] ), erros, label="Erros")
plt.title('Erros Real - Calculado')
plt.xlabel('Dia')
plt.ylabel('Valor R$')
plt.legend()
plt.grid()
plt.show(block=True)