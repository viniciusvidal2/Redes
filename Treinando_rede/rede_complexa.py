
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization

# O script principal chamaria aqui a construcao da rede sequencial totalmente conectada, com um numero de entradas e saidas la definido
class Rede_complexa:
    @staticmethod
    def montar(int_tempo, features, saidas):
        
        kernel=int(int_tempo/2)
        
        rede = Sequential()

        rede.add(Conv1D(filters=40, kernel_size=kernel, activation='relu', input_shape=(int_tempo, features)))
        #rede.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        
        #rede.add(Conv1D(filters=40, kernel_size=10, activation='relu'))
        #rede.add(BatchNormalization())

        #rede.add(MaxPooling1D(pool_size=2))
        
        rede.add(Conv1D(filters=40, kernel_size=2, activation='relu'))

        #rede.add(Conv1D(filters=40, kernel_size=2, activation='relu'))

        rede.add(TimeDistributed(Flatten()))
        rede.add(LSTM(30))

        rede.add(Dense(10, activation = 'relu'))

        #rede.add(Dropout(0.25))

        #rede.add(Dense(saidas*15, activation = 'elu'))

        rede.add(Dropout(0.50))
        
        rede.add(Dense(saidas, activation = 'sigmoid'))

        return rede