
import datetime, shelve
import collections
import bovespa
import numpy as np
import Indicadores as Ind
from sklearn.externals import joblib


def concatena(A,b):
    for i in range(len(A)):
        A[i].append(b[i])

    return A

def readFromBovespa(filepath,filename,stockname,year,Ndays):
    """
    filepath = Caminho completo para o aquivo baixado do site da B3
        Ex: 'E:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos\\

    filename = Nome do que deseja salvar o arquivo.
        Ex: Para o nome COTACAO_, o aquivo será salvo como COTACAO_stockname_year.txt

    stockname = Vetor de strings com o nome dos ativos que se deseja ler e salvar do arquivo baixado na B3
        Ex: stockname = ['ABCP11','BOVA11','ITUB4','VALE4','PETR4','TIET11','SANB11','ELET6','ABEV3','GOAU4','CMIG4']

    year = Vetor de inteiros com os anos dos ativos que se deseja ler e salvar do arquivo baixado na B3
        Ex: year = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

    Ndays = Número de dias para o calculo dos indicadores
        Ex: Ndays = 14 (a maioria dos indicadores utiliza 14 como parâmetro)

    Salva Arquivo com os dados Lidos da B3 no formato 
    [Data, Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D)]
    onde:
    MME = Média Móvel Exponencial
    IFR = Indice de Força Relativo
    OBV = On balance Volume
    OS  = Oscilador Estocastico
   

    """
    for i in range(len(year)):
       bf = bovespa.File(filepath+'\\COTAHIST_A'+str(year[i])+'.TXT') # Grin

       for j in range(len(stockname)):

           print('\n ATIVO: {} \t ANO: {} '.format(stockname[j],year[i]))
           inputData = []
           try:

               for rec in bf.query(stock=stockname[j]):
                   print('Data:{} \t Abertura:{} \t  Fechamento:{} '.format(str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close))
                   #Preenchendo o vetor com os indicadores
                   inputData.append([str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close,rec.price_high,rec.price_low,rec.price_mean,rec.volume])

               # Calculo dos indicadores e concatenação com a matriz inputdata para que possa ser salva em um arquivo
               Array = np.array(inputData)

               Med = Ind.MME(Ndays,Array[:,2])
               Ifr = Ind.IFR(Ndays,Array[:,1:3])
               Obv = Ind.OBV(Array[:,1:7])
               K,D = Ind.OS(Ndays,Array[:,1:7])

               V1 = concatena(inputData,Med)
               V2 = concatena(V1,Ifr)
               V3 = concatena(V2,Obv)
               V4 = concatena(V3,K)
               V5 = concatena(V4,D)
           
               inputData = np.array(V5)

               # salva Dados em arquivo já descriptografado (B3) com adição de indicadores

               joblib.dump(inputData,filename+stockname[j]+'_'+str(year[i])+'.txt')

           except:
                print("{}_{}  NÃO ENCONTRADO! ".format(stockname[j], year[i]))





