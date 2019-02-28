from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout

# O script principal chamaria aqui a construcao da rede sequencial totalmente conectada, com um numero de entradas e saidas la definido
class Rede_recursiva:
    @staticmethod
    def montar(int_tempo, features, saidas):
        rede = Sequential()
        #rede.add(TimeDistributed(Conv1D(filters=10, kernel_size=3, activation='relu'), input_shape=(None, int_tempo, features)))
        #rede.add(TimeDistributed(Flatten()))
        #rede.add(LSTM(6, activation='relu'))
        rede.add(LSTM(100, input_shape=(int_tempo, features)))
        rede.add(Dense(15, activation='relu'))
        rede.add(Dense(30, activation='relu'))
        rede.add(Dense(70, activation='relu'))
        rede.add(Dropout(0.5))
        rede.add(Dense(10, activation='relu'))
        rede.add(Dropout(0.5))
        rede.add(Dense(saidas, activation='sigmoid'))

        return rede