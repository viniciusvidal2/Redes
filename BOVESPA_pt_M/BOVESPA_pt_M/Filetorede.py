
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer 

from datetime import datetime

#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go

#plotly.tools.set_credentials_file(username='mathausfsilva', api_key='201113510277')
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
    ## This function read the data for the file in the filePath variable

        dados = joblib.load(filePath)

        Data_array = np.array(dados)

        return Data_array


def OrganizeData(Data_array, sampleSize, inputNumber,Ndays,outPosition):
    """
    This function organize the Data_array considering the sampleSize and inputNumber
    Data_array  --> Array organized as : Rows = Days  Columns = Day Inputs 
    sampleSize  --> Number of samples  (days in this case)
    inputNumber --> Number of inputs for each day
    Ndays       --> Numero de dias a ser previsto
    outPosition --> index of each output inside de Data_array
   
    Organiza os dados para 50 dias com 6 entradas em cada dia [Abertura, Fechamanto, Maxima, Miníma, Média, Volume]
    """
    OutputData   = []
    OutputResult = []

    for i in range(len(Data_array) - (sampleSize+Ndays)): 
        VecData  = []
        VecData2 = []

        for j in range(sampleSize):
            for k in range(inputNumber):
                VecData.append(Data_array[i+j][k])
        OutputData.append(VecData)

        for l in range(Ndays):
            for m in range(len(outPosition)):
                VecData2.append(Data_array[i+j+l+1][outPosition[m]])

        OutputResult.append(VecData2)
          
  # Retorna as Entrada e as Saida da Rede
    return OutputData, OutputResult


def Normalize(array):
    """
    Vetor deve ter a entrada do tipo  [Data, Abertura, Fechamento, Maxima, Mínima, Média, Volume, MME, IFR, OBV, OS(K,D),...]
    Normaliza os valores entre 0 e 1 e remove a primeira coluna do aquivo com dados (primeira coluna  = Data)

    Todas as colunas da matriz são normalizadas com excessão da primeira.
    """
    maximo = []
    minimo = [] 
    Amp    = []

    Array  = np.array(array[:,1:len(array[1])])
    Array  = Array.astype('float32')
    ArrayT = np.transpose(Array)

    # Initialization of the output array
    Aux_arr = ArrayT

    # Save the maximun and minimun of each array
    for i in range(len(ArrayT)):
        maximo.append(np.max(ArrayT[i,:]))
        minimo.append(np.min(ArrayT[i,:]))

        x = maximo[i] - minimo[i]
        Amp.append(x)

    # Normalize the array considering each variable range
    for i in range(len(ArrayT)):
        Aux_arr[i,:] = (ArrayT[i,:] - minimo[i])/Amp[i]

    Arr = np.transpose(Aux_arr)

    return Arr

