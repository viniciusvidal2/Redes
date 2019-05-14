#################################################################################################################################################################################
####################################################################### Rede de previsao para ACOES #############################################################################
##### Autores: Vinicius, Mathaus e Guilherme (talvez) #####
##### Primeira Utilizacao marcada: 15/05/2019         #####
#################################################################################################################################################################################

##### ----------------------------------------------------------------------- INCLUDES ------------------------------------------------------------------------------------ #####
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
from rede_complexa import Rede_complexa
import ferramentas_leitura as fl

import numpy as np
from numpy import array
import scipy.io as sio

from sklearn.model_selection import train_test_split

##### --------------------------------------------------------- Leitura dos arquivos de acao passados -------------------------------------------------------------------- #####
# Nome e caminho para a acao, dias da semana dos ultimos fechamentos (a partir dai sera previsto futuro)
acao = 'VVAR3'
dias = ['SEX', 'SEG']
filePath = ['./../DATAYEAR/ultimoCOTACAO_'+acao+'_2019.txt']

# Locais e nomes de arquivos salvos
pasta_salvar    = 'Melhores_redes/'
rede_rec_salvar = pasta_salvar+'atual_rec.hdf5'
rede_com_salvar = pasta_salvar+'atual_com.hdf5'
arq_rec         = pasta_salvar+'arquitetura_rec.png'
arq_com         = pasta_salvar+'arquitetura_com.png'
load_rec        = pasta_salvar+'atual_rec.hdf5'
load_com        = pasta_salvar+'atual_com.hdf5'
efi_rec         = pasta_salvar+'eficacia_atual_recursiva.mat'  
efi_com         = pasta_salvar+'eficacia_atual_complexa.mat' 

# Se quisermos treinar as redes e necessario chavear a flag, caso quiser so executar colocar FALSE
FLAG_TREINAMENTO = True
redes_a_executar = 80 # 1 para recursiva, 2 para complexa, maior que 10 para todas (assim podem entrar novas redes) 

# Traduzindo o arquivo e organizando os dados
SampleSize = 10
Dias_previstos = 1
Dias_futuros   = 3

# Organizando dados
print('\nOrganizando dados...')
Data = fl.ReadData(filePath)
Data = fl.removeDate(Data)
Data_norm, maximos, minimos, amplitudes = fl.Normalize(Data)

variacao  = fl.ROC( Dias_futuros, Data[:,1]) # Calculo a variacao no valor do fechamento real, normalizado para 'DIAS FUTUROS' anteriores - PENSAR MELHOR
variacaof = fl.ROCF(Dias_futuros, Data[:,1]) # Calculo para a previsao (SAIDA) no valor do fechamento real daqui a 'DIAS FUTUROS', normalizado sobre o dia atual
Data_norm = fl.concatena(Data_norm, variacao ) # Adiciono como coluna no vetor de dados em geral, posso utilizar para saida ou entrada
Data_norm = fl.concatena(Data_norm, variacaof) # Adiciono como coluna no vetor de dados em geral, posso utilizar para saida ou entrada

mascara_entradas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0] # variacao futura nao usada (ROCF), somente ROC
InputNumber = sum(mascara_entradas) # Quantas entradas existirao
OutputPositions = np.array([12]) # variaveis a serem previstas - olhar dentro da funcao, porem aqui somente a ROCF sera prevista (variacao futura)
X, Y = fl.OrganizeData(Data_norm, SampleSize, InputNumber, Dias_previstos, OutputPositions, mascara_entradas)

X = X.reshape(X.shape[0], SampleSize, InputNumber)
Y = Y.reshape(Y.shape[0], Y.shape[1])

X_desconhecido = X[-Dias_futuros+1:] # Separa os dias que estao no final das amostras, nao tem previsao e serao previstos na realidade - TESTE REAL

for i in range(1, Dias_futuros): # Remove entao os mesmos do final da lista de forma iterativa para nao treinar com valores errados (contem 0 em ROCF)
    X = np.delete(X, -1, axis=0)
    Y = np.delete(Y, -1, axis=0)

# Conjuntos de treino, para ficar organizado, para X e Y, e que sera dividido em treinamento e validacao segundo fracionamento definido
Xtreino = X.copy()
Ytreino = Y.copy()
(trainx, valx, trainy, valy) = train_test_split(Xtreino, Ytreino, test_size=0.25, random_state=10)


##### -------------------------------------------------------------- Treinar as redes predefinidas ----------------------------------------------------------------------- #####

if FLAG_TREINAMENTO:
    epocas = 10000 # por quantas epocas treinar

    # Otimizador ADAM para treinar a rede
    print("\nCriando otimizador e rede...")
    adam = Adam(lr=0.0005, decay=0.000005, amsgrad=False)

    ##### ----- Montar as redes propostas ----- #####
    rede_rec = Rede_recursiva.montar(SampleSize, InputNumber, Dias_previstos*len(OutputPositions))
    rede_com = Rede_complexa.montar(SampleSize, InputNumber, Dias_previstos*len(OutputPositions))

    if redes_a_executar == 1 or redes_a_executar >= 10: # Rede recursiva
        # Compilando a rede
        rede_rec.compile(optimizer=adam, loss='mse')
        # Callbacks para salvar melhor rede e parar treino antes
        melhor_rede = ModelCheckpoint(rede_rec_salvar, save_best_only=True, verbose=1, monitor='val_loss')
        parada_forcada = EarlyStopping(monitor='val_loss', patience=800, verbose=1)
        # Plotando arquitetura da rede
        plot_model(rede_rec, arq_rec, show_shapes=True, show_layer_names=True)
        # Aqui acontece o treino e ajuste de pesos realmente - observar BATCH SIZE
        print("\n\nComecando o treinamento da rede RECURSIVA ...")
        H_rec = rede_rec.fit(trainx, trainy, validation_data=(valx, valy), batch_size=10, epochs=epocas, callbacks=[melhor_rede, parada_forcada], verbose=2)
    
    if redes_a_executar == 2 or redes_a_executar >= 10: # Rede complexa
        # Compilando a rede
        rede_com.compile(optimizer=adam, loss='mse')
        # Callbacks para salvar melhor rede e parar treino antes
        melhor_rede = ModelCheckpoint(rede_com_salvar, save_best_only=True, verbose=1, monitor='val_loss')
        parada_forcada = EarlyStopping(monitor='val_loss', patience=800, verbose=1)
        # Plotando arquitetura da rede
        plot_model(rede_com, arq_com, show_shapes=True, show_layer_names=True)
        # Aqui acontece o treino e ajuste de pesos realmente - observar BATCH SIZE
        print("\n\nComecando o treinamento da rede COMPLEXA ...")
        H_com = rede_com.fit(trainx, trainy, validation_data=(valx, valy), batch_size=10, epochs=epocas, callbacks=[melhor_rede, parada_forcada], verbose=2)

 
##### ------------------------------------------------------- Executar sobre os dados a serem previstos ------------------------------------------------------------------ #####
# Leitura e teste das melhores redes recentemente criadas
if redes_a_executar == 1 or redes_a_executar >= 10:
    rede2_rec = load_model(load_rec)
    saida_teste_rec = rede2_rec.predict(X_desconhecido, verbose=1) # Testando sobre o conjunto desconhecido de qualquer forma

if redes_a_executar == 2 or redes_a_executar >= 10:
    rede2_com = load_model(load_com)
    saida_teste_com = rede2_com.predict(X_desconhecido, verbose=1) # Testando sobre o conjunto desconhecido de qualquer forma


if FLAG_TREINAMENTO: # Avaliar sobre o conjunto de validacao caso as redes acadarem de ser treinadas
    
    if redes_a_executar == 1 or redes_a_executar >= 10:
        saida_val_rec = rede2_rec.predict(valx, verbose=1) # Testando sobre o conjunto de validacao
        
        ##### ----- Calculo de efetividade da tendencia na saida ----- #####
        tendencia_real = []
        tendencia_calc = []
        acertos = 0

        for amostra in valy:
            tendencia_real.append(sum(amostra))
        for amostra in saida_val_rec:
            tendencia_calc.append(sum(amostra))

        for i in range(len(tendencia_calc)):
            if tendencia_calc[i] > 0 and tendencia_real[i] > 0 or tendencia_calc[i] < 0 and tendencia_real[i] < 0:
                acertos = acertos + 1

        eficacia_rec = acertos/len(tendencia_calc) * 100
        # Salvar a acuracia em uma variavel .mat
        sio.savemat(efi_rec, {'eficacia': eficacia_rec})

    if redes_a_executar == 2 or redes_a_executar >= 10:
        saida_val_com = rede2_com.predict(valx, verbose=1) # Testando sobre o conjunto de validacao
        
        ##### ----- Calculo de efetividade da tendencia na saida ----- #####
        tendencia_real = []
        tendencia_calc = []
        acertos = 0

        for amostra in valy:
            tendencia_real.append(sum(amostra))
        for amostra in saida_val_com:
            tendencia_calc.append(sum(amostra))

        for i in range(len(tendencia_calc)):
            if tendencia_calc[i] > 0 and tendencia_real[i] > 0 or tendencia_calc[i] < 0 and tendencia_real[i] < 0:
                acertos = acertos + 1

        eficacia_com = acertos/len(tendencia_calc) * 100
        # Salvar a acuracia em uma variavel .mat
        sio.savemat(efi_com, {'eficacia': eficacia_com})


##### ------------------------------------------------------------------ Print dos resultados ---------------------------------------------------------------------------- #####

# Para o treinamento, acuracias, caso contrario estamos interessados nos valores percentuais e finais das acoes no futuro
if FLAG_TREINAMENTO:

    if redes_a_executar == 1 or redes_a_executar >= 10:
        print("\nACURACIA rede RECURSIVA: %.2f pct\n"%(eficacia_rec))
    if redes_a_executar == 2 or redes_a_executar >= 10:
        print("\nACURACIA rede COMPLEXA : %.2f pct\n"%(eficacia_com))

# Percentual e final para cada dia previsto 
print("\nAcao atual: %s\n\n"%(acao))

if redes_a_executar == 1 or redes_a_executar >= 10:
    porcentagem = (np.ones(saida_teste_rec.shape)+saida_teste_rec)
    efi = sio.loadmat(efi_rec)
    print("\nREDE RECURSIVA eficacia %.2f - Teste:\n"%(efi['eficacia'][0]))
    for i in range(len(dias)):
        valor = float(Data[-len(dias)+i][1])*float(porcentagem[i][0]) # Fechamento do ultimo dia de cada conjunto amostral de entrada
        print("Previsao para %s: %.2f pct\t%.2f reais\n"%( fl.return_DoW(dias[i]), 100*saida_teste_rec[i][0], valor ))
    
if redes_a_executar == 2 or redes_a_executar >= 10:
    porcentagem = (np.ones(saida_teste_com.shape)+saida_teste_com)
    efi = sio.loadmat(efi_com)
    print("\nREDE COMPLEXA eficacia %.2f - Teste:\n"%(efi['eficacia'][0]))
    for i in range(len(dias)):
        valor = float(Data[-len(dias)+i][1])*float(porcentagem[i][0]) # Fechamento do ultimo dia de cada conjunto amostral de entrada
        print("Previsao para %s: %.2f pct\t%.2f reais\n"%( fl.return_DoW(dias[i]), 100*saida_teste_com[i][0], valor ))


##### ------------------------------------------------------------------ Plot dos resultados ---------------------------------------------------------------------------- #####

if FLAG_TREINAMENTO:

    if redes_a_executar == 1 or redes_a_executar >= 10:
        # # Evolucao do treinamento da rede
        # plt.figure()
        # plt.style.use("ggplot")
        # plt.plot(np.arange(0, len(H_rec.history["loss"])), H_rec.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, len(H_rec.history["val_loss"])), H_rec.history["val_loss"], label="val_loss")
        # plt.title('Evolution RECURSIVA')
        # plt.xlabel('Epoch #')
        # plt.ylabel('Value')
        # plt.legend()
        # Saida sobre o conjunto de validacao
        plt.figure()
        plt.plot(np.arange(0, len(valy         )), 100*np.array(valy         ), 'b+', label="Variacoes reais")
        plt.plot(np.arange(0, len(saida_val_rec)), 100*np.array(saida_val_rec), 'mo', label="Variacoes calculadas")
        plt.title('Variacoes - Rede RECURSIVA')
        plt.xlabel('Dia')
        plt.ylabel('Percentual dia anterior')
        plt.legend()
        plt.grid()

    if redes_a_executar == 2 or redes_a_executar >= 10:
        # # Evolucao do treinamento da rede
        # plt.figure()
        # plt.style.use("ggplot")
        # plt.plot(np.arange(0, len(H_rec.history["loss"])), H_rec.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, len(H_rec.history["val_loss"])), H_rec.history["val_loss"], label="val_loss")
        # plt.title('Evolution COMPLEXA')
        # plt.xlabel('Epoch #')
        # plt.ylabel('Value')
        # plt.legend()
        # Saida sobre o conjunto de validacao
        plt.figure()
        plt.plot(np.arange(0, len(valy         )), 100*np.array(valy         ), 'b+', label="Variacoes reais")
        plt.plot(np.arange(0, len(saida_val_com)), 100*np.array(saida_val_com), 'mo', label="Variacoes calculadas")
        plt.title('Variacoes - Rede RECURSIVA')
        plt.xlabel('Dia')
        plt.ylabel('Percentual dia anterior')
        plt.legend()
        plt.grid()

    plt.show(block=True)
