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


def descript():

    # === Leitura dos dados no arquivo baixado na bolsa pra descriptografar ===

    # Descriptografia dos dados baixados da B3

    filepath  = 'E:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos'
    filename  = 'COTACAO_'
    stockname = ['BOVA11','VALE4','ABCP11','ITUB4','PETR4','TIET11','SANB11','ELET6','ABEV3','GOAU4','CMIG4','VVAR3']
    year      = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
    path_save = 'C:\\Users\\Mathaus\\Documents\\GitHub\\Redes\\DATAYEAR\\'

    Ndays     = 14

    Rd.readFromBovespa(filepath,filename,stockname,year,Ndays,path_save)

def leituradado():

    # ======== Leitura e organização dos dados para entrada na rede ===========

    filePath    = 'COTACAO_ELET6_2017.txt'
    SampleSize  = 3
    InputNumber = 1
    NdaysPrev = 1

    # Faz a leitura do arquivo
    Array = Fr.ReadData(filePath)

    #Array = Fr.Normalize(Data)

    # Calcula todos o indices para conferir com o que está guardado no arquivo.
    med = Ind.MME(14,Array[:,2])
    ifr = Ind.IFR(14,Array[:,1:3])
    obv = Ind.OBV(Array[:,1:7])
    K,D = Ind.OS(14,Array[:,1:7])
    R   = Ind.williams_R(14,Array[:,1:7])

    Variacao = Ind.ROC(30,Array[:,2])

    # Criação de Figura para a comparação dos valores já lidos no aquivo, com os aqui calculados
    # Figura com indicadores salvos previamente

    f, figure = plt.subplots(4,sharex=True)

    figure[0].plot(np.transpose(Array[:,7].astype('float32')),label = 'MME')
    figure[0].grid(1)

    figure[1].plot(np.transpose(Array[:,8].astype('float32')),label = 'IFR')
    figure[1].grid(1)

    figure[2].grid(1)
    figure[2].plot(np.transpose(Array[:,9].astype('float32')),label = 'OBV')

    figure[3].grid(1)
    figure[3].plot(np.transpose(Array[:,10].astype('float32')),label = 'K')
    figure[3].plot(np.transpose(Array[:,11].astype('float32')),label = 'D')


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
    
    plt.show()


#[Data, Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D)]

descript()