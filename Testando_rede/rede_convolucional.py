from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense

class Rede_convolucional:
    @staticmethod
    def montar(dias, variaveis, saidas):
        inputshape = (dias, variaveis, 1)
        kernel = (4, 4)
        rede = Sequential()

        # aqui ja montando as camadas da rede, esperando uma "imagem" com "dias" linhas e
        # "variaveis" colunas
        rede.add(Conv2D(20, kernel, padding="same", input_shape=inputshape))
        rede.add(Activation('elu'))
        rede.add(Conv2D(20, kernel, padding="same", input_shape=inputshape))
        rede.add(Activation('elu'))

        rede.add(Flatten())
        rede.add(Dropout(0.25))
        rede.add(Dense(10))
        rede.add(Dense(saidas))
        
        return rede



