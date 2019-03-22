
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
        
        kernel=int(5)
        
        rede = Sequential()

        rede.add(LSTM(20, return_sequences=True, activation='tanh', input_shape=(int_tempo, features), dropout = 0))

        #rede.add(LSTM(10, return_sequences=True, activation='tanh', input_shape=(int_tempo, features), dropout = 0))
        
        #rede.add(LSTM(5, return_sequences=True, activation='tanh', input_shape=(int_tempo, features), dropout = 0))

        rede.add(Conv1D(filters=20, kernel_size=kernel, activation='tanh'))
        #rede.add(Dropout(0.25))
        #rede.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        
        #rede.add(Conv1D(filters=40, kernel_size=10, activation='relu'))
        #rede.add(BatchNormalization()
        
        #rede.add(Conv1D(filters=40, kernel_size=2, activation='relu'))

        #rede.add(Conv1D(filters=40, kernel_size=2, activation='relu'))
        #rede.add(MaxPooling1D(pool_size=2))
        
        #rede.add(Flatten())
        #rede.add(LSTM(30))

        #rede.add(Dense(100, activation = 'relu'))

        #rede.add(Dropout(0.25))

        #rede.add(Dense(20, activation = 'relu'))

        #rede.add(Dropout(0.50))

        rede.add(LSTM(saidas))
        
        #rede.add(Dense(saidas, activation = 'tanh'))

        return rede