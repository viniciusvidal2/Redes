
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

# O script principal chamaria aqui a construcao da rede sequencial totalmente conectada, com um numero de entradas e saidas la definido
class Rede_complexa:
    @staticmethod
    def montar(subseq, int_tempo, features, saidas):
        kernel=5
        rede = Sequential()
        rede.add(TimeDistributed(Conv1D(filters=64, kernel_size=kernel, activation='relu'), 
                                  input_shape=(None, int(int_tempo/subseq), features)))
        #rede.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        
        rede.add(TimeDistributed(Flatten()))
        rede.add(LSTM(30, activation = 'relu'))

        rede.add(Dense(saidas*5, activation = 'relu'))

        rede.add(Dropout(0.25))
        
        rede.add(Dense(saidas, activation = 'relu'))

        return rede