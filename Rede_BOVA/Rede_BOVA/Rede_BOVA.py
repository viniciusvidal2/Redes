
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/

import pandas
import datetime, shelve
import collections
import bovespa
import numpy as np
from sklearn.externals import joblib


## CODIGO PARA A LEITURA DE DADOS UTILIZANDO O JOBLIB
#filename = 'historic_BOVA11_2018.TXT'
#dados = joblib.load(filename)



## DADOS DA AÇÃO
filename  = 'COTACAO_'
stockname =['ABCP11','BOVA11','ITUB4','VALE4','PETR4','TIET11','SANB11','ELET6','ABEV3','GOAU4','CMIG4']
YEAR = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

#bf = bovespa.File('E:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos\\COTAHIST_A2018.TXT')# casa 

for i in range(len(YEAR)):
       bf = bovespa.File('D:\\GoogleDrive\\Python_BOLSA_DE_Valores\\Dados_Historicos\\COTAHIST_A'+str(YEAR[i])+'.TXT') # Grin

       for j in range(len(stockname)):

           print('\n ATIVO: {} \t ANO: {} '.format(stockname[j],YEAR[i]))
           inputData = []


           for rec in bf.query(stock=stockname[j]):
               print('Data:{}   Abertura:{}   Fechamento:{}   Maxímo:{}   Mínimo:{}   Média:{}'.format(str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close,rec.price_high,rec.price_low,rec.price_mean))
               inputData.append([str(rec.date.day)+'-'+str(rec.date.month)+'-'+str(rec.date.year),rec.price_open, rec.price_close,rec.price_high,rec.price_low,rec.price_mean,rec.volume])
        
           joblib.dump(inputData,filename+stockname[j]+'_'+str(YEAR[i])+'.txt')




#print(len(inputData))

