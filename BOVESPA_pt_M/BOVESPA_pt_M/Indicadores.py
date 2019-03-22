

import numpy as np

def MME(Ndias,Data):
    """
    A entrada deve ser um vetor apenas com os dados que se deseja a média móvel.
    Diferentemente das outras funções onde a entrada é toda a matriz de dados

    Média Móvel Exponencial
    Curto Prazo: de 5 a 20 períodos
    Médio Prazo: de 20 a 60 períodos
    Longo Prazo: mais de 100 períodos]
    N = Número de dias ( ciclos)

    o primeiro MME[anterior] deve ser a média aritimética dos dias anteriores em um dado
    periodo de tempo (ou pode ser assumido o primeiro valor de preço).

    Sinal de Compra: a MME mais curta cruza para cima da MME mais longa
    Sinal de Venda: a MME mais curta cruza para baixo da MME mais longa

    """
    
    MME = np.zeros(len(Data))
    Data = Data.astype('float32')

    # Constante de dias
    K = 2/(Ndias+1);

    # primeira média é aritimética
    MME[Ndias-1] = sum(Data[0:Ndias])/Ndias

    for i in range(len(Data)-Ndias):
        # Media movel exponencial 
        MME[Ndias+i] = (Data[Ndias+i]- MME[Ndias+i-1])*K + MME[Ndias+i-1]

    return MME

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
         
        ROC[N+i] = (Fechamento[N+i] - Fechamento[i]) / Fechamento[i]

    return ROC

def IFR(Ndias,Data):
    """
   Data must be ---> [Abertura, Fechamanto, Maxima, Miníma, Média, Volume] 
   Ndias = Numeros de dias no intervalo

   Índice de Força Relativa (IFR) - bom para curto prazo

   U = Média das cotações dos últimos n dias em que a cotação da ação subiu. Trata-se da soma das cotações dos últimos n dias em que a cotação da ação subiu, dividido por n.
   D = Média das cotações dos últimos n dias em que a cotação da ação caiu. Trata-se da soma das cotações dos últimos n dias em que a cotação da ação caiu, dividido por n.
   n = O numero de dias mais utilizado pelo mercado é 14, e recomendado por Wilder quando da publicação de seu livro.  Mas também é comum usar um IFR de 9 ou 25 dias, 
   e você pode customizar o indicador para quantos períodos desejar.
     
   Região de Sobre-compra: >70 ---- VENDER
   Região de Sobre-venda : <30 ---- COMPRAR
    
   """

    Data = Data.astype('float32')
    #Nmax = 0
    Valmax = 0
    Valmin = 0
    i = 0
    IFR = np.zeros(len(Data[:,1]))

    while i < len(Data[:,1]) - Ndias:

        for j in range(0,Ndias,1):

            if(Data[i+j,1] < Data[i+j+1,1]):
                Valmax = Valmax + (Data[i+j+1,1]-Data[i+j,1])
            elif Data[i+j,1] > Data[i+j+1,1]:
                    Valmin = Valmin + (Data[i+j,1]-Data[i+j+1,1])
        
        U = Valmax/Ndias
        D = Valmin/Ndias

        IFR[j+i] = 100 - (100/(1+(U/D)))
        Valmax = 0
        Valmin = 0
        i = i + 1
    return IFR


def OBV(Data):
    """
    Data must be ---> [Abertura, Fechamanto, Maxima, Miníma, Média, Volume]

    On Balance Volume (OBV)
     – Medir a força da tendência,
     – Identificar possíveis reversões
     – Sinalizar o surgimento de novas tendências
    """

    Data = Data.astype('float32')
    OBV  = np.zeros(len(Data[:,1]))
    i = 1
    while i < (len(Data[:,1])):
        if Data[i,1] > Data[i,0]:
            OBV[i] = OBV[i-1] + Data[i,5]
        elif Data[i,1] < Data[i,0]:
            OBV[i] = OBV[i-1] - Data[i,5]
        else:
            OBV[i] = OBV[i-1]
        i = i + 1

    return OBV


def OS(Ndias, Data):
    """

    Ndias = geralmente igual a 14
    Oscilador Estocastico
    Data must be ---> [Abertura, Fechamanto, Maxima, Miníma, Média, Volume,...]

    Se a linha %K cruzar para acima da %D temos, em geral a configuração de um call de compra.
    Por outro lado, se a %K cruzar para baixo da %D temos um sinal de venda.

    Outro recurso seria observar o comportamento da %K em relação ao nível dos 50 pontos do Oscilador Estocástico.
    Em geral, Quando a linha %K cruza acima de 50 é dado um sinal de compra, alternativamente, 
    quando o %K cruza para baixo dos 50 pode-se configurar uma oportunidade de venda.

    """

    dias = 3
    Data = Data.astype('float32')
    K  = np.zeros(len(Data[:,1]))
    D  = np.zeros(len(Data[:,1]))
    i = 0
    while i < (len(Data[:,1])):
        minima = min(Data[i:i+Ndias,1])
        maxima = max(Data[i:i+Ndias,1])
        if(maxima - minima)!= 0:
            K[i] = (Data[i,1] - minima) / (maxima - minima)*100
        else:
            K[i] = 0

        if i>=dias:
            D[i] = sum(K[i-dias:i])/dias
        else:
            D[i] = K[i]
       
        i = i + 1

    return K,D


def williams_R(Ndias,Data):
    """
     Data must be --> [Abertura, Fechamanto, Maxima, Miníma, Média, Volume]

    O indicador Williams %R geralmente é utilizado pela maioria dos analistas como o período de 14 dias.

    O indicador produz valores que respeitam uma escala de -100 a 0. Quando o indicador chega à faixa de 0 a -20,
    é um indicativo de que a ação está na região de sobrecompra, já que o preço atingiu um valor próximo ao mínimo (mathaus-maximo) observado no período n.
    Por outro lado, quando o indicador atinge a faixa de -80 a -100, considera-se que ele atingiu a região sobrevendida,
    já que o preço está próximo ao máximo registrado no período n observado.
     
      0  -- -20  --> VENDE
      80 -- -100 --> COMPRA
    """
       
    Data = Data.astype('float32')
    R  = np.zeros(len(Data[:,1]))
    i = 0
    while i < (len(Data[:,1])):
        minima = min(Data[i:i+Ndias,1])
        maxima = max(Data[i:i+Ndias,1])
        if(maxima - minima)!= 0:
            R[i] = (maxima - Data[i,1]) / (maxima - minima )* -100
        else:
            R[i] = 0
               
        i = i + 1

    return R


def plotCandle(Ndias,Data1):
    """

       Data must be ---> [Data, Abertura, Fechamanto, Maxima, Miníma, Média, Volume]
    """
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

    trace = go.Candlestick(x=dates,
                            open=open_data,
                            high=high_data,
                            low=low_data,
                            close=close_data)
    data = [trace]
    plotly.offline.plot(data, filename='candlestick_datetime.html')
    py.iplot(data, filename='candlestick_datetime.html')


