from keras.models import Sequential
from keras.models import load_model

from keras.layers.core import Dense
from keras.layers import LSTM

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

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
filePath = ['./../DATAYEAR/COTACAO_PETR4_2015.txt', './../DATAYEAR/COTACAO_PETR4_2016.txt', './../DATAYEAR/COTACAO_PETR4_2017.txt', './../DATAYEAR/COTACAO_PETR4_2018.txt']
#filePath = ['./../DATAYEAR/COTACAO_BOVA11_2015.txt', './../DATAYEAR/COTACAO_BOVA11_2016.txt', './../DATAYEAR/COTACAO_BOVA11_2017.txt', './../DATAYEAR/COTACAO_BOVA11_2018.txt']
SampleSize = 30
Subsequences = 1
mascara_entradas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
InputNumber = sum(mascara_entradas)
Dias_previstos = 5
OutputPositions = np.array([1]) # variaveis a serem previstas - olhar dentro da funcao

# Organizando dados
print('Organizando dados...')
Data = fl.ReadData(filePath)
Data_norm, maximos, minimos, amplitudes = fl.Normalize(Data)

X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions, mascara_entradas)

X = X.reshape(X.shape[0], SampleSize, InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])

# Uma vez feito o reshape e mais facil pegar o valor de fechamento e calcular 
## Plotar aqui as saidas de fechamento e abertura em um grafico para analise visual de padroes
#plt.figure()
#plt.plot(np.arange(0, Y.shape[0]), Y[:, 0], label="Fechamento real")
#plt.title('Dados Reais')
#plt.xlabel('Dia')
#plt.ylabel('Valor normalizado')
#plt.legend()
#plt.show(block=True)

#### funciona aqui a separacao entre treino, validacao e teste automatica apos o reshape mesmo
#(trainx, val_testx, trainy, val_testy) = train_test_split(X        , Y        , test_size=0.3, random_state=20)
#(valx  , testx    , valy  , testy    ) = train_test_split(val_testx, val_testy, test_size=0.1, random_state=30)

########################## Aqui separa os ultimos dias_teste dias para teste, porem o treino e validacao mantem sendo aleatorios no restante das amostras
intervalo_teste = np.arange(-10, -1) # epoca ainda desconhecida, amostras sequenciais
X_teste, Y_teste = X[intervalo_teste].copy(), Y[intervalo_teste].copy()

dias_teste = np.arange(0, X_teste.shape[0], Dias_previstos) # testar sobre as amostras de teste em um intervalo de Dias_previstos
(testx, testy) = X_teste[dias_teste], Y_teste[dias_teste] # Dias que realmente havera teste sobre, entrarao na funcao predict

Xtreino = np.delete(X, X.shape[0]+intervalo_teste, axis=0)
Ytreino = np.delete(Y, Y.shape[0]+intervalo_teste, axis=0)
#Xtreino = np.delete(X, intervalo_teste+X.shape[0], axis=0)
#Ytreino = np.delete(Y, intervalo_teste+X.shape[0], axis=0)
(trainx, valx, trainy, valy) = train_test_split(Xtreino, Ytreino, test_size=0.10, random_state=30)

########################### Aqui monta a rede usando a classe desejada
treinar = False
if treinar:
    epocas = 2000 # por quantas epocas treinar

    print("Criando otimizador e rede...")
    adam = Adam(lr=0.0005, decay=0.000001, amsgrad=False)
    #rede = Rede_convolucional.montar(SampleSize, InputNumber, len(Out_posi))
    rede = Rede_recursiva.montar(SampleSize, InputNumber, Dias_previstos*len(OutputPositions))
    #rede = Rede_complexa.montar(SampleSize, InputNumber, 1)
    rede.compile(optimizer=adam, loss='mse')
    # Callbacks para salvar melhor rede e parar treino antes
    melhor_rede = ModelCheckpoint("Melhores_redes/atual.hdf5", save_best_only=True, verbose=1, monitor='val_loss')
    parada_forcada = EarlyStopping(monitor='val_loss', patience=40, verbose=1)
    # Plotando arquitetura da rede
    plot_model(rede, "Melhores_redes/arquitetura_atual.png", show_shapes=True, show_layer_names=True)

    # Aqui acontece o treino e ajuste de pesos realmente - observar BATCH SIZE
    print("Comecando o treinamento da rede...")
    H = rede.fit(trainx, trainy, validation_data=(valx, valy), batch_size=2, epochs=epocas, callbacks=[melhor_rede], verbose=1)

########################### Testar em cima da melhor rede possivel salva anteriormente
rede2 = load_model('Melhores_redes/atual.hdf5')
testx=valx
testy=valy
saida_teste = rede2.predict(testx, verbose=1)

########################### Plot grafico da evolucao da rede
if treinar:
    plt.style.use("ggplot")
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
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

#testey_par[conjunto_par]          = np.zeros([len(conjunto_par), 1])
#testey_impar[conjunto_impar]      = np.zeros([len(conjunto_impar), 1])
#saida_teste_par[conjunto_par]     = np.zeros([len(conjunto_par), 1])
#saida_teste_impar[conjunto_impar] = np.zeros([len(conjunto_impar), 1])

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
testey_par2        = fl.ReturnRealValue(testey_par, minimos, maximos, amplitudes, OutputPositions)
testey_impar2      = fl.ReturnRealValue(testey_impar, minimos, maximos, amplitudes, OutputPositions)
saida_teste_par2   = fl.ReturnRealValue(saida_teste_par, minimos, maximos, amplitudes, OutputPositions)
saida_teste_impar2 = fl.ReturnRealValue(saida_teste_impar, minimos, maximos, amplitudes, OutputPositions)

plt.figure()
plt.plot(np.arange(0, Y_real.shape[0] )         , Y_real          , label="Fechamento Real")
plt.plot(np.arange(0, saida_teste_real.shape[0]), saida_teste_real, label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor R$')
plt.legend()
plt.grid()

plt.figure()
plt.plot(np.arange(0, testey_par2.shape[0]), testey_par2, 'b+', label="Fechamento Real")
plt.plot(np.arange(0, testey_impar2.shape[0]), testey_impar2, 'k+', label="Fechamento Real")
plt.plot(np.arange(0, saida_teste_par2.shape[0]), saida_teste_par2, 'ro', label="Fechamento Calculado")
plt.plot(np.arange(0, saida_teste_impar2.shape[0]), saida_teste_impar2, 'mo', label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor R$')
plt.legend()
plt.grid()

 #Calculo do erro entre os valores reais R$
erros = Y_real[0:len(saida_teste_real)] - saida_teste_real
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