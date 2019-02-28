
# http://www.bmfbovespa.com.br/pt_br/servicos/market-data/historico/mercado-a-vista/cotacoes-historicas/

from tkinter import *
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


def OrganizeData(Data_array, sampleSize, inputNumber,Ndays,outPosition):

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

        for l in range(Ndays):
            for m in range(len(outPosition)):
                VecData2.append(Data_array[i+j+l+1][outPosition[m]])

        OutputResult.append(VecData2)

        

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

### CODIGO PARA A LEITURA DE DADOS UTILIZANDO O JOBLIB
#filePath    = 'COTACAO_BOVA11_2011.txt'
#SampleSize  = 3
#InputNumber = 1
#Ndays = 1
#Out_posi = np.array([0,1])

#Data = ReadData(filePath)
#Out_Data = Normalize(Data)
#Output = OrganizeData(Out_Data, SampleSize, InputNumber,Ndays,Out_posi)
#Output = np.array(Output)
#Output = Output.astype('float32')

#print("FIM!! numero de linhas: {}".format(len(OutputData)))


######################## functions
def printName(event):
    print("Mathaus")

def leftClick(event):
    print("Left")

def middleClick(event):
    print("Middle")

def rightClick(event):
    print("Right")



root = Tk()

############################## Frame


frame = Frame(root,width=300,height=250)

frame.bind("<Button-1>", leftClick)
frame.bind("<Button-2>", middleClick)
frame.bind("<Button-3>", rightClick)
frame.pack()

#topFrame = Frame(root)
#topFrame.pack()

#bottomFrame = Frame(root)
#bottomFrame.pack(side=BOTTOM)

########################################################## Buttons
button2 = Button(frame,text="Cancel",fg="silver")
button1 = Button(frame,text="Login",fg="red")
button1.bind("<Button-1>",printName)
#button1 = Button(root,text="Login",fg="red",command=printName)


#button3 = Button(bottomFrame,text="Button 3",fg="blue")

#button1.pack(side=LEFT)
#button2.pack(side=LEFT)
#button3.pack(side=BOTTOM)

########################################################## labels
#thelabel = Label(root,text="Home Page")
#thelabel.pack()
#three = Label(root,text="Three",bg="blue", fg="white" )
#three.pack(side=LEFT, fill=Y)

label_1 = Label(frame,text="Name")
label_2 = Label(frame,text="Password")

######################################################### Entry
entry_1 = Entry(frame,text="Name")
entry_2 = Entry(frame,text="Password")


######### checkbox

c = Checkbutton(frame,text="Keep me logged in")
c.grid(row=2,column=1,columnspan=2)


# Organizando os componentes

label_1.grid(row=0,sticky=E)
label_2.grid(row=1,sticky=E)

entry_1.grid(row=0,column=1)
entry_2.grid(row=1,column=1)

button1.grid(row=3,column=1,sticky=E)
button2.grid(row=3,column=1,sticky=W)





root.mainloop()
 

