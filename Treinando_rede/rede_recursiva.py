from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

# O script principal chamaria aqui a construcao da rede sequencial totalmente conectada, com um numero de entradas e saidas la definido
class Rede_recursiva:
    @staticmethod
    def montar(int_tempo, features, saidas):
        rede = Sequential()
    
        rede.add(LSTM(40, input_shape=(int_tempo, features)))

        rede.add(Dense(35, activation='relu'))

        #rede.add(BatchNormalization())
        rede.add(Dense(30, activation='sigmoid'))

        rede.add(Dense(30, activation='sigmoid'))
        #rede.add(Dropout(0.5))

        rede.add(Dense(20, activation='relu'))
        rede.add(Dropout(0.25))

        rede.add(Dense(saidas, activation='relu'))

        return rede