from keras.models import load_model

from keras.utils import plot_model
from keras.utils import np_utils
import matplotlib.pyplot as plt

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
filePath = ['../DATAYEAR/COTACAO_BOVA11_2018.txt']
SampleSize = 50
Subsequences = 2
InputNumber = 6
Dias_previstos = 7
OutputPositions = np.array([1]) # variaveis a serem previstas - olhar dentro da funcao

# Organizando dados
print('Organizando dados...')
Data = fl.ReadData(filePath)
Data_norm, maximos, minimos, amplitudes = fl.Normalize(Data)
X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions)

X = X.reshape(X.shape[0], Subsequences, int(SampleSize/Subsequences), InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])

dias_teste = np.arange(0, X.shape[0], 7)
X, Y = X[dias_teste], Y[dias_teste]

########################### Testar em cima da melhor rede possivel salva anteriormente
print('Rodando a rede...')
rede2 = load_model('../Treinando_rede/Melhores_redes/atual.hdf5')
saida_teste = rede2.predict(X, verbose=1)

########################### Trazer de volta para valores reais
Y_real           = fl.ReturnRealValue(Y          , minimos, maximos, amplitudes, OutputPositions)
saida_teste_real = fl.ReturnRealValue(saida_teste, minimos, maximos, amplitudes, OutputPositions)

########################### AVALIACAO QUANTITATIVA
#erros_raw  = Y_real - saida_teste_real
#erros_abs  = np.abs(Y_real - saida_teste_real)
#erro_medio = np.mean(erros_abs, axis=0)
#erro_max   = np.max(erros_raw, axis=0)
#erro_min   = np.min(erros_raw, axis=0)

#desvio_padrao = np.std(erros_abs, axis=0)

#erros_pct  = erros_raw/Y_real
#erros_abs_pct = np.abs(erros_pct)
#erro_medio_pct = np.mean(erros_abs_pct, axis=0)
#erro_max_pct   = np.max(erros_pct, axis=0)
#erro_min_pct   = np.min(erros_pct, axis=0)

#desvio_padrao_pct = np.std(erros_abs_pct, axis=0)

#print("\nABERTURA:\n\n")
#print("Erros em reais:\n")
#print('medio: {}\t maximo: {}\t minimo: {}\nDesvio padrao: {}'.format(erro_medio[0], erro_max[0], erro_min[0], desvio_padrao[0]))
#print("\n\n\nFECHAMENTO:\n\n")
#print("Erros em reais:\n")
#print('Erro medio: {}\tErro maximo: {}\t Erro_minimo: {}\nDesvio padrao: {}'.format(erro_medio[1], erro_max[1], erro_min[1], desvio_padrao[1]))

#print("\nABERTURA:\n\n")
#print("Erros em pct:\n")
#print('medio: {}\t maximo: {}\t minimo: {}\nDesvio padrao: {}'.format(erro_medio_pct[0], erro_max_pct[0], erro_min_pct[0], desvio_padrao_pct[0]))
#print("\n\n\nFECHAMENTO:\n\n")
#print("Erros em pct:\n")
#print('Erro medio: {}\tErro maximo: {}\t Erro_minimo: {}\nDesvio padrao: {}'.format(erro_medio_pct[1], erro_max_pct[1], erro_min_pct[1], desvio_padrao_pct[1]))


########################### Plot para comparacao com o conjunto de teste - Precos de saida de fechamento
Y_real = Y_real.flatten()
saida_teste_real = saida_teste_real.flatten()

plt.figure()
plt.plot(np.arange(0, Y_real.shape[0] ), Y_real , label="Fechamento Real")
plt.plot(np.arange(0, saida_teste_real.shape[0]), saida_teste_real, label="Fechamento Calculado")
plt.title('Fechamento')
plt.xlabel('Dia')
plt.ylabel('Valor_norm')
plt.legend()
plt.grid()
plt.show(block=True)
