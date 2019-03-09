
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/

import pandas
import datetime, shelve
from datetime import datetime
import collections
import bovespa
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import Normalizer 

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
            for k in range(len(masc_ent)):
                if masc_ent[k] == 1:
                    VecData.append(Data_array[i+j][k])
        OutputData.append(VecData)

        for l in range(Ndays):
            for m in range(len(outPosition)):
                VecData2.append(Data_array[i+j+l+1][outPosition[m]])

        # Dessa forma ficam todos os dias de uma variavel por vez agrupados em cada amostra
        # Exemplo: [a1 a2 a3 b1 b2 b3] para variaveis a e b, por 3 dias
        #for m in range(len(outPosition)):
        #    for l in range(Ndays):
        #        if i+j+l+1 < len(Data_array):
        #            VecData2.append(Data_array[i+j+l+1][outPosition[m]])
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