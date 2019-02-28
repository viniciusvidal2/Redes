
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/

import pandas
import datetime, shelve
import collections
import bovespa
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer 

def ReadData(filePath):
    ## This function read the data for the file in the filePath variable

        dados = joblib.load(filePath)

        Data_array = np.array(dados)

        return Data_array


def OrganizeData(Data_array, sampleSize, inputNumber, Ndays, outPosition):

    # This function organize the Data_array considering the sampleSize and inputNumber
    # sampleSize --> Number of samples  (days in this case)
    # inputNumber --> Number of inputs for each day
    # Data_array --> Array organized as : Rows = Days  Columns = Day Inputs 
    # Ndays --> Numero de dias a ser previsto
    # outPosition --> index of each output inside de Data_array

    # Organiza os dados para 50 dias com 6 entradas em cada dia [Abertura, Fechamanto, Maxima, Miníma, Média, Volume]

    OutputData  =  []
    OutputResult = []

    for i in range(len(Data_array) - (sampleSize+Ndays)): 
        VecData = []
        VecData2 = []
        for j in range(sampleSize):
            for k in range(inputNumber):
                VecData.append(Data_array[i+j][k])
        OutputData.append(VecData)

        #for l in range(Ndays):
        #    for m in range(len(outPosition)):
        #        VecData2.append(Data_array[i+j+l+1][outPosition[m]])

        # Dessa forma ficam todos os dias de uma variavel por vez agrupados em cada amostra
        # Exemplo: [a1 a2 a3 b1 b2 b3] para variaveis a e b, por 3 dias
        for m in range(len(outPosition)):
            for l in range(Ndays):
                if i+j+l+1 < len(Data_array):
                    VecData2.append(Data_array[i+j+l+1][outPosition[m]])
        OutputResult.append(VecData2)

    OutputData   = np.array(OutputData)
    OutputResult = np.array(OutputResult)

    OutputData   = OutputData.astype('float32')
    OutputResult = OutputResult.astype('float32')

    return OutputData, OutputResult


def Normalize(array):

    maximo = []
    minimo = [] 
    Amp    = []

    Array  = np.array(array[:,1:len(array[1])])
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

    return Arr