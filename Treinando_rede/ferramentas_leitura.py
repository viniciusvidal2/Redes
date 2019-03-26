
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/

import pandas
import datetime, shelve
from datetime import datetime
import collections
import bovespa
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer 

def removeDate(A):
    
    A = A[:,1:len(A[1])]
    
    return A

def plotCandle(Ndias,Data1):

        #[data, Abertura, Fechamanto, Maxima, Miníma, Média, Volume]
    date = Data1[:,0]
    Data = Data1[:,1:7].astype('float32')

    dia = np.zeros(len(date))
    mes = np.zeros(len(date))
    ano = np.zeros(len(date))

    dates = []

    for i in range(len(date)):
        dia[i] = str(date[i]).split("-")[0]
        mes[i] = str(date[i]).split("-")[1]
        ano[i] = str(date[i]).split("-")[2]

        dates.append(datetime(year=int(ano[i]), month=int(mes[i]),day=int(dia[i])))

    
    open_data  = Data[:,0]
    close_data = Data[:,1]
    high_data  = Data[:,2]
    low_data   = Data[:,3]

    #R  = np.zeros(len(Data[:,1]))

    #open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    #high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    #low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    #close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    #dates = [datetime(year=2013, month=10, day=10),
    #         datetime(year=2013, month=11, day=10),
    #         datetime(year=2013, month=12, day=10),
    #         datetime(year=2014, month=1, day=10),
    #         datetime(year=2014, month=2, day=10)]

    trace = go.Candlestick(x=dates,
                           open=open_data,
                           high=high_data,
                           low=low_data,
                           close=close_data)
    data = [trace]
    plotly.offline.plot(data, filename='candlestick_datetime.html')
    py.iplot(data, filename='candlestick_datetime.html')

def ReadData(filePath):
    ## This function reads the data for the file in the filePath variable
    Dados_array = []
    for file in filePath:
        dados_temp = joblib.load(file)
        Dados_array.extend(dados_temp)
        
    Data_array = np.array(Dados_array)
    return Data_array


def OrganizeData(Data_array, sampleSize, inputNumber, Ndays, outPosition, masc_ent):

    # This function organize the Data_array considering the sampleSize and inputNumber
    # sampleSize --> Number of samples  (days in this case)
    # inputNumber --> Number of inputs for each day
    # Data_array --> Array organized as : Rows = Days  Columns = Day Inputs 
    # Ndays --> Numero de dias a ser previsto
    # outPosition --> index of each output inside de Data_array

    OutputData  =  []
    OutputResult = []

    for i in range(len(Data_array) - (sampleSize+Ndays)): 
        VecData = []
        VecData2 = []
        for j in range(sampleSize):
            for k in range(len(masc_ent)): # todas as entradas possiveis, mas so valem mesmo as marcadas com 1 na mascara
                if masc_ent[k] == 1:
                    VecData.append(Data_array[i+j][k])
        OutputData.append(VecData)

        for l in range(Ndays):
            for m in range(len(outPosition)):
                VecData2.append(Data_array[i+j+l+1][outPosition[m]])

        OutputResult.append(VecData2)

    OutputData   = np.array(OutputData)
    OutputResult = np.array(OutputResult)

    OutputData   = OutputData.astype('float32')
    OutputResult = OutputResult.astype('float32')

    # Retorna as Entrada e as Saida da Rede
    return OutputData, OutputResult


def Normalize(Array):

    maximo = []
    minimo = [] 
    Amp    = []

    #Array  = np.array(array[:,1:len(array[1])])
    Array = Array.astype('float32')
    ArrayT = np.transpose(Array)

    # Initialization of the output array
    Aux_arr = ArrayT

    # Save the maximun and minimun of each array
    for i in range(len(ArrayT)):
        maximo.append(np.max(ArrayT[i,:]))
        minimo.append(np.min(ArrayT[i,:]))

        x = maximo[i] - minimo[i]
        Amp.append (x)

    # Normalize the array considering each variable range
    for i in range(len(ArrayT)):
        Aux_arr[i,:] = (ArrayT[i,:] - minimo[i])/Amp[i]

    Arr = np.transpose(Aux_arr)

    return Arr, maximo, minimo, Amp

def ReturnRealValue(y_norm, mins, maxs, amps, variables):
    # This function receives normalized data from the network output and its ranges, returning 
    # the real values back to the user
    
    y = y_norm
    #bring back every variable of interest from the real ranges
    for v in range(len(variables)):
        index = variables[v]
        y = mins[index] + amps[index]*y_norm

    return y

def concatena(A,b):
    """
    A = [Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D),...]
    """


    A = A.tolist();
    for i in range(len(A)):
        A[i].append(b[i])
    
    A = np.array(A)
    A = A.astype('float32')
    return A

def ROC(N,Fechamento):
    """
    Rate of Change (ROC) – Taxa de Variação    
    SAÍDA de -1 (zero) a (1 Um)

    -1 --- 0 Variação negativa
     0 --- 1 Variação positiva

    N    = numero de dias anteriores no  qual se deseja a ROC.
    Data = [valor1, valor2, valor3, valor4, valor5, ...]

    A entrada deve ser um vetor apenas com os dados que se deseja a taxa de variação
    Diferentemente das outras funções onde a entrada é toda a matriz de dados

    Descrição Matemática

    Taxa de Variação[i] = (Preço de Fechamento[i] – Preço de Fechamento[i-n]) / Preço de Fechamento[i-n])

    o primeiro MME[anterior] deve ser a média aritimética dos dias anteriores em um dado
    periodo de tempo (ou pode ser assumido o primeiro valor de preço).

    Sinal de Compra: a MME mais curta cruza para cima da MME mais longa
    Sinal de Venda: a MME mais curta cruza para baixo da MME mais longa

    """
    
    ROC = np.zeros(len(Fechamento))
    Fechamento = Fechamento.astype('float32')

    for i in range(len(Fechamento)-N):
         
        ROC[i] = (Fechamento[N+i] - Fechamento[i]) / Fechamento[i]

    return ROC