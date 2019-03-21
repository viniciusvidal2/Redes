from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers.merge import concatenate


# O script principal chamaria aqui a construcao da rede sequencial totalmente conectada, com um numero de entradas e saidas la definido
class Rede_concatenada:
    @staticmethod
    def montar(variaveis, saidas):
        # Feito na mao, mas existirao tantas redes paralelas quanto o numero de variaveis
        # a principio serao 6
        low    = Sequential()
        high   = Sequential()
        mean   = Sequential()
        volume = Sequential()
        open   = Sequential()
        close  = Sequential()

        # Valor de baixa - low
        low.add(Dense(30, input_shape=(50, )))
        low.add(Dense(10))
        # Valor de baixa - high
        high.add(Dense(30, input_shape=(50, )))
        high.add(Dense(10))
        # Valor de baixa - mean
        mean.add(Dense(30, input_shape=(50, )))
        mean.add(Dense(10))
        # Valor de baixa - volume
        volume.add(Dense(30, input_shape=(50, )))
        volume.add(Dense(10))
        # Valor de baixa - open
        open.add(Dense(30, input_shape=(50, )))
        open.add(Dense(10))
        # Valor de baixa - close
        close.add(Dense(30, input_shape=(50, )))
        close.add(Dense(10))

        #merged = Concatenate([low, high, mean, volume, open, close])
        merged = concatenate([low, high, mean, volume, open, close])
        densa
        final = Dense(1, activation='relu')(merged) # Aqui adicionando a ultima camada ao que foi mergido anteriormente
        
        return final

        