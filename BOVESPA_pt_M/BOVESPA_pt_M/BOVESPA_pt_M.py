"""
Programa desenvolvido com a inteção de prever os balores das cotações da Bolsa de Valores
Os arquivos utilizados com os históricos 
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/


"""

import matplotlib.pyplot as plt
import numpy as np

import Bovespatofile as Rd
import Indicadores as Ind
import Filetorede as Fr

def concatena(A,b):
    """
    ENTRADA DE DADOS
    A = Data must be --> [Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D),...]
    b = Vetor coluna a ser concatenado 
    """
    if type(A) is not list:
        A = A.tolist()

    for i in range(len(A)):
        A[i].append(b[i])
    
    A = np.array(A)
    A = A.astype('float32')
    return A


def decrypt():

    # === Leitura dos dados no arquivo baixado na bolsa pra descriptografar ===

    # Descriptografia dos dados baixados da B3

    filepath  = 'E:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos'
    filename  = 'COTACAO_'
    stockname = ['BOVA11','USIM5','FLRY3','TRPL4','VVAR3','PETR4','TIET11','ELET6']
    year      = [2019]
    path_save = 'C:\\Users\\Mathaus\\Documents\\GitHub\\Redes\\DATAYEAR'
    
    # Número de dias para o calculo dos indicadores
    Ndays     = 14

    Rd.readFromBovespa(filepath,filename,stockname,year,Ndays,path_save)

def leituradado():

    # ======== Leitura e organização dos dados para entrada na rede ===========

    filePath    = 'COTACAO_FLRY3_2019.txt'
    SampleSize  = 3
    InputNumber = 1
    NdaysPrev   = 1

    # Faz a leitura do arquivo
    Array = Fr.ReadData(filePath)

    # remover  a coluna com a data para trabalhar apenas com os numeros
    Array = Fr.removeDate(Array)

    Array2 = Fr.Normalize(Array)

    # Calcula todos o indices para conferir com o que está guardado no arquivo.
    med = Ind.MME(14,Array[:,1])
    ifr = Ind.IFR(14,Array[:,0:2])
    obv = Ind.OBV(Array[:,0:6])
    K,D = Ind.OS(14,Array[:,0:6])
    R   = Ind.williams_R(14,Array[:,0:6])


    Variacao = Ind.ROC(1,Array[:,1])
    # O Arquivo lido é um array e deve ser convertido para lista para ser concatenado
    Array = concatena(Array,Variacao)

    # Criação de Figura para a comparação dos valores já lidos no aquivo, com os aqui calculados
    # Figura com indicadores salvos previamente

    f, figure = plt.subplots(4,sharex=True)

    figure[0].plot(np.transpose(Array[:,6].astype('float32')),label = 'MME')
    figure[0].grid(1)

    figure[1].plot(np.transpose(Array[:,7].astype('float32')),label = 'IFR')
    figure[1].grid(1)

    figure[2].grid(1)
    figure[2].plot(np.transpose(Array[:,8].astype('float32')),label = 'OBV')

    figure[3].grid(1)
    figure[3].plot(np.transpose(Array[:,9].astype('float32')),label = 'K')
    figure[3].plot(np.transpose(Array[:,10].astype('float32')),label = 'D')


    # Figura com indicadores calculados aqui
    f, figure1 = plt.subplots(4,sharex=True)

    figure1[0].plot(med,label = 'MME')
    figure1[0].grid(1)

    figure1[1].plot(ifr,label = 'IFR')
    figure1[1].grid(1)

    figure1[2].grid(1)
    figure1[2].plot(obv,label = 'OBV')

    figure1[3].grid(1)
    figure1[3].plot(K,label = 'K')
    figure1[3].plot(D,label = 'D')

     # Figura com Variação percentual calculados


    f, figure2 = plt.subplots(2,sharex=True)

    figure2[0].plot(Variacao,label = 'ROC')
    figure2[0].grid(1)

    figure2[1].plot(np.transpose(Array[:,11].astype('float32')),label = 'ROC')
    figure2[1].grid(1)
    
    plt.show()


#[Data, Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D)]



a = [1, 2, 3]
b = [4, 5, 6]
C = a + b
print (C)

decrypt()
#leituradado()
