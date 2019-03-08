
import datetime, shelve
import collections
import bovespa
import numpy as np
import Indicadores as Ind
from sklearn.externals import joblib


def readFromBovespa(filepath,filename,stockname,year):
    """
    Salva Arquivo com os dados Lidos da B3 no formato 
    [Data, Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D)]
    onde:
    MME = Média Móvel Exponencial
    IFR = Indice de Força Relativo
    OBV = On balance Volume
    OS  = Oscilador Estocastico
    

    filepath = Caminho completo para o aquivo baixado do site da B3
        Ex: 'E:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos\\

    filename = Nome do que deseja salvar o arquivo.
        Ex: Para o nome COTACAO_, o aquivo será salvo como COTACAO_stockname_year.txt

    stockname = Vetor de strings com o nome dos ativos que se deseja ler e salvar do arquivo baixado na B3
        Ex: stockname = ['ABCP11','BOVA11','ITUB4','VALE4','PETR4','TIET11','SANB11','ELET6','ABEV3','GOAU4','CMIG4']

    year = Vetor de inteiros com os anos dos ativos que se deseja ler e salvar do arquivo baixado na B3
        Ex: year = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]


    """
    for i in range(len(year)):
       bf = bovespa.File(filepath+'\\COTAHIST_A'+str(year[i])+'.TXT') # Grin

       for j in range(len(stockname)):

           print('\n ATIVO: {} \t ANO: {} '.format(stockname[j],year[i]))
           inputData = []

           for rec in bf.query(stock=stockname[j]):
               print('Data:{} \t Abertura:{} \t  Fechamento:{} '.format(str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close))
               #Preenchendo o vetor com os indicadores
               inputData.append([str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close,rec.price_high,rec.price_low,rec.price_mean,rec.volume])

           joblib.dump(inputData,filename+stockname[j]+'_'+str(year[i])+'.txt')

