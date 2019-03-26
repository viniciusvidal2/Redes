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
#filePath = ['./../DATAYEAR/COTACAO_CMIG4_2018.txt', './../DATAYEAR/COTACAO_CMIG4_2019.txt']
#filePath = ['./../DATAYEAR/COTACAO_VVAR3_2012.txt', './../DATAYEAR/COTACAO_VVAR3_2013.txt', './../DATAYEAR/COTACAO_VVAR3_2014.txt',
#            './../DATAYEAR/COTACAO_VVAR3_2015.txt', './../DATAYEAR/COTACAO_VVAR3_2016.txt', './../DATAYEAR/COTACAO_VVAR3_2017.txt', './../DATAYEAR/COTACAO_VVAR3_2018.txt']
#filePath = ['./../DATAYEAR/COTACAO_ITUB4_2011.txt', './../DATAYEAR/COTACAO_ITUB4_2012.txt', './../DATAYEAR/COTACAO_ITUB4_2013.txt', './../DATAYEAR/COTACAO_ITUB4_2014.txt',
#            './../DATAYEAR/COTACAO_ITUB4_2015.txt', './../DATAYEAR/COTACAO_ITUB4_2016.txt', './../DATAYEAR/COTACAO_ITUB4_2017.txt', './../DATAYEAR/COTACAO_ITUB4_2018.txt']
filePath = ['./../DATAYEAR/COTACAO_PETR4_2016.txt', './../DATAYEAR/COTACAO_PETR4_2017.txt', './../DATAYEAR/COTACAO_PETR4_2018.txt', './../DATAYEAR/COTACAO_PETR4_2019.txt']
#filePath = ['./../DATAYEAR/COTACAO_BOVA11_2016.txt', './../DATAYEAR/COTACAO_BOVA11_2017.txt', './../DATAYEAR/COTACAO_BOVA11_2018.txt', './../DATAYEAR/COTACAO_BOVA11_2019.txt']
SampleSize = 15
Dias_previstos = 3

# Organizando dados
print('Organizando dados...')
Data = fl.ReadData(filePath)
Data = fl.removeDate(Data)
Data_norm, maximos, minimos, amplitudes = fl.Normalize(Data)

variacao  = fl.ROC(1, Data[:,1]) # Calculo a variacao no valor do fechamento real, nao normalizado
Data_norm = fl.concatena(Data_norm, variacao) # Adiciono como coluna no vetor de dados em geral, posso utilizar para saida ou entrada

mascara_entradas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
InputNumber = sum(mascara_entradas)
OutputPositions = np.array([11]) # variaveis a serem previstas - olhar dentro da funcao
X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions, mascara_entradas)

X = X.reshape(X.shape[0], SampleSize, InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])

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
intervalo_teste = np.arange(-9, -1) # epoca ainda desconhecida, amostras sequenciais
X_teste, Y_teste = X[intervalo_teste].copy(), Y[intervalo_teste].copy()

dias_teste = np.arange(0, X_teste.shape[0], Dias_previstos) # testar sobre as amostras de teste em um intervalo de Dias_previstos
(testx, testy) = X_teste[dias_teste], Y_teste[dias_teste] # Dias que realmente havera teste sobre, entrarao na funcao predict

Xtreino = np.delete(X, X.shape[0]+intervalo_teste, axis=0)
Ytreino = np.delete(Y, Y.shape[0]+intervalo_teste, axis=0)
#Xtreino = np.delete(X, intervalo_teste+X.shape[0], axis=0)
#Ytreino = np.delete(Y, intervalo_teste+X.shape[0], axis=0)
(trainx, valx, trainy, valy) = train_test_split(Xtreino, Ytreino, test_size=0.25, random_state=30)

########################### Aqui monta a rede usando a classe desejada
treinar = True
if treinar:
    epocas = 10000 # por quantas epocas treinar

    print("Criando otimizador e rede...")
    adam = Adam(lr=0.001, decay=0.0000005, amsgrad=False)
    #rede = Rede_convolucional.montar(SampleSize, InputNumber, len(Out_posi))
    #rede = Rede_recursiva.montar(SampleSize, InputNumber, Dias_previstos*len(OutputPositions))
    rede = Rede_complexa.montar(SampleSize, InputNumber, Dias_previstos*len(OutputPositions))
    rede.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    # Callbacks para salvar melhor rede e parar treino antes
    melhor_rede = ModelCheckpoint("Melhores_redes/atual.hdf5", save_best_only=True, verbose=1, monitor='val_loss')
    parada_forcada = EarlyStopping(monitor='val_loss', patience=400, verbose=1)
    # Plotando arquitetura da rede
    plot_model(rede, "Melhores_redes/arquitetura_atual.png", show_shapes=True, show_layer_names=True)

    # Aqui acontece o treino e ajuste de pesos realmente - observar BATCH SIZE
    print("Comecando o treinamento da rede...")
    #rede = load_model('Melhores_redes/atual.hdf5')
    H = rede.fit(trainx, trainy, validation_data=(valx, valy), batch_size=100, epochs=epocas, callbacks=[melhor_rede, parada_forcada], verbose=2)

########################### Testar em cima da melhor rede possivel salva anteriormente
rede2 = load_model('Melhores_redes/atual.hdf5')
testx=valx.copy()
testy=valy.copy()
saida_teste = rede2.predict(testx, verbose=1)

########################### Calculo de acuracia em termos de tendencia
tendencia_real = []
tendencia_calc = []
acertos = 0
flag_calculo = 2 # 1 para valores direto na saida, 2 para diferenca entre os dias
# Quando valores direto na saida
if flag_calculo == 1:
    for amostra in testy:
        if amostra[-1]-amostra[0] > 0:
            tendencia_real.append(1)
        elif amostra[-1]-amostra[0] == 0:
            tendencia_real.append(0)
        else:
            tendencia_real.append(-1)

    for amostra in saida_teste:
        if amostra[-1]-amostra[0] > 0:
            tendencia_calc.append(1)
        elif amostra[-1]-amostra[0] < 0:
            tendencia_calc.append(-1)
        else:
            tendencia_calc.append(0)
            
    for i in range(len(tendencia_calc)):
        if tendencia_calc[i] == tendencia_real[i]:
            acertos = acertos + 1

    zeros_real = tendencia_real.count(0)
    zeros_calc = tendencia_calc.count(0)
# Quando valores diferenciais na saida
elif flag_calculo == 2:
    for amostra in testy:
        tendencia_real.append(sum(amostra))
    for amostra in saida_teste:
        tendencia_calc.append(sum(amostra))

    for i in range(len(tendencia_calc)):
        if tendencia_calc[i] > 0 and tendencia_real[i] > 0 or tendencia_calc[i] < 0 and tendencia_real[i] < 0:
            acertos = acertos + 1
    
acuracia = acertos/len(tendencia_calc) * 100
print("\n\nACURACIA: {}%\n\n".format(acuracia))

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

testey_par[conjunto_par]          = np.zeros([conjunto_par.size  , testey_par.shape[1]]  )
testey_impar[conjunto_impar]      = np.zeros([conjunto_impar.size, testey_impar.shape[1]])
saida_teste_par[conjunto_par]     = np.zeros([conjunto_par.size  , testey_par.shape[1]]  )
saida_teste_impar[conjunto_impar] = np.zeros([conjunto_impar.size, testey_impar.shape[1]])

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

############################ Trazer de volta para valores reais
if flag_calculo == 1:
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

elif flag_calculo == 2:
    plt.figure()
    plt.plot(np.arange(0, len(tendencia_real)), 100*np.array(tendencia_real), 'b+', label="Variacoes reais")
    plt.plot(np.arange(0, len(tendencia_calc)), 100*np.array(tendencia_calc), 'mo', label="Variacoes calculadas")
    plt.title('Variacoes')
    plt.xlabel('Dia')
    plt.ylabel('Percentual dia anterior')
    plt.legend()
    plt.grid()
plt.show(block=True)