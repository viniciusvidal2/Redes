from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Dense

class Rede_convolucional:
    @staticmethod
    def montar(dias, variaveis, saidas):
        inputshape = (dias, variaveis, 1)
        kernel = int(dias/3)

        # aqui ja montando as camadas da rede, esperando um array com features para cada variavel
        # dados devem ser organizados na entrada da forma:
        # [samples, timesteps, features]
        rede = Sequential()

        rede.add(Conv1D(filters=64, kernel_size=kernel, input_shape=(dias, variaveis)))
        rede.add(Activation('relu'))
        rede.add(BatchNormalization())
        
        rede.add(Conv1D(filters=32, kernel_size=kernel))
        rede.add(Activation('relu'))
        rede.add(Dropout(0.1))
        rede.add(Flatten())
        
        rede.add(Dense(10, activation='relu')) 
        rede.add(Dropout(0.25))
        
        rede.add(Dense(saidas, activation='relu'))
        
        return rede



