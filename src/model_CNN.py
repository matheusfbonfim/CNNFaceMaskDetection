"""
CNN-Face-Mask-Detection.ipynb

Original file is located at
    https://colab.research.google.com/drive/1UCvyHJ-dbCfVyzztk6uch3Pf6ZD-9VTm?usp=sharing
"""
## =====================================
## [0] IMPORTANDO BIBLIOTECAS

import numpy as np
# Sequential model
from keras.models import Sequential
# Bibliotecas - Camadas
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# Funcão de custo
from keras.metrics import categorical_crossentropy
# Pre-Processamento de imagens
from keras.preprocessing.image import ImageDataGenerator
# P/ salvar o modelo
from keras.callbacks import ModelCheckpoint
# Matriz de confusão
from sklearn.metrics import confusion_matrix
# Plotar o modelo
from keras.utils.vis_utils import plot_model

import itertools
import os
import matplotlib.pyplot as plt


## =====================================
## =====================================
## [1] PRÉ-PROCESSAMENTO DE DADOS DE IMAGEM

#   Dataset: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset
    #   Organizaremos nossos dados em conjuntos de treinamento, validação e teste -> Manualmente
    #   Movendo subconjuntos de dados para subdiretórios - Conjunto de dados separado.

# Caminhos para os dados de treinamento, validação e teste
path_train = '../database/New Masks Dataset/Train'        # Caminho - Train
path_valid = '../database/New Masks Dataset/Test'         # Caminho - Validação
path_test = '../database/New Masks Dataset/Validation'    # Caminho - Teste


## =====================================
## [1.1] Gerando o conjunto de dados de treinamento, validação/teste

## ---------------------------------------
## [1.1.1] ImageDataGenerator

# Usaremos a funcionalidade Image Data Generator de Keras para lidar com nossos dados de entrada
# Contribui substituindo as imagens de treinamento originais por versões aumentadas das mesmas,
# enquanto o modelo é treinado nelas.


# shear_range - significa que a imagem será distorcida ao longo de um eixo
# zoom_range - Amplia aleatoriamente a imagem e adiciona novos valores de pixel ao redor da imagem
# ou interpola valores de pixel respectivamente
train_gen = ImageDataGenerator (rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_gen = ImageDataGenerator(rescale = 1./255)
test_gen  = ImageDataGenerator(rescale = 1./255)

## ---------------------------------------
##  [1.1.2] Flow_from_directory

# Chama-se o fluxo de diretorio, no qual é passado os dados reais e especificado
# como queremos que esses dados sejam processados

train_batches = train_gen.flow_from_directory(directory=path_train, batch_size=32, color_mode='rgb',
                                              target_size=(200, 200), seed = 42, class_mode='binary')

valid_batches = valid_gen.flow_from_directory(directory=path_valid, batch_size=32, color_mode='rgb',
                                              target_size=(200, 200), seed = 42, class_mode='binary')

test_batches = test_gen.flow_from_directory(directory=path_test, batch_size=32, color_mode='rgb',
                                            target_size=(200, 200), seed = 42, class_mode='binary')


# Vendo os rotulos e os dados gerados
print(train_batches.class_indices)  # Rede neural precisa de indices categoricos em formato inteiro


## =====================================
## [1.2] Verificando as imagens

# Verificação em tempo de execucao da quantidade de imagens 
#assert train_batches.n == 600
#assert valid_batches.n == 306
#assert test_batches.n == 100
#assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# Pegando um unico lote de imagens e as labels  do train_batches
images, labels = next(train_batches)  # Deve haver 32 imagens (Tamanho do batch)


# Plotando as imagens do lote acima
# Esta função irá traçar imagens na forma de uma grade com 1 linha e 10 colunas onde as imagens são colocadas
def plotingImages(images_array):
    fig, axes = plt.subplots(4, 8, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_array, axes):
        ax.imshow(img)
        ax.axis('off')  # Desliga os eixos em torno de cada imagem

    plt.tight_layout()
    plt.show()


plotingImages(images)  # Visualizado apenas 64 imagens 
print(labels)


## =====================================
## =====================================
## [2] MODELO - REDE NEURAL CONVOLUCIONAL

images.shape    # Os dados de treinamento são imagens de pessoas: 64 amostras, cada uma com 200 por 200 pixels,
                # e com ultima dimensão de 3 canais devido ao RGB

## =====================================
## [2.1] Construindo o modelo

## INICIANDO O MODELO

# Modelo como uma pilha simples de camadas onde cada camada tem exatamente um tensor de entrada e um tensor de saída .
model = Sequential()

##---------------------------------------
## 1 - Inserindo a camada convolucional

# A primeira camada está conectada a todos os pixels na imagem de entrada
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same' ,input_shape=(200, 200, 3)))
# MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))


##---------------------------------------
## 2 - Inserindo outra camada convolucional

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
# MaxPooling
model.add(MaxPooling2D(pool_size=(2, 2)))

##---------------------------------------
## Flatten
model.add(Flatten())
model.add(Dropout(0.5))

##---------------------------------------
## MLP
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Adicionando uma outra camada oculta
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))

##---------------------------------------
# Saida da rede é uma camada totalmente conectada com uma unidade para cada classe: 2 tipos
  # Usa a função de ativação softmax para decidir qual das tres classes foi apresentada
    # Nos dá a probabilidade para cada saída correspondente
model.add(Dense(2, activation='softmax'))


## =====================================
### [2.2] Visualizando o modelo

##---------------------------------------
# summarize model
model.summary()

##---------------------------------------
# Plotando o modelo
plot_model(model, to_file='../model/plot_model.png', show_shapes=True, show_layer_names=True)


## =====================================
### [2.3] Compilando o modelo

# Compilamos o modelo
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## =====================================
### [2.4] Treinando e validando o modelo

# O retorno de chamada é usado em conjunto com o treinamento model.fit()para salvar um modelo
# ou pesos (em um arquivo de ponto de verificação) em algum intervalo, para que o modelo ou pesos
# possam ser carregados posteriormente para continuar o treinamento a partir do estado salvo.
model_save = ModelCheckpoint('../model/face-mask-detection.h5', monitor = 'val_loss', verbose = 0, save_best_only=True, mode = 'auto')

##---------------------------------------
model.fit(x=train_batches, validation_data=valid_batches, epochs = 20, callbacks=[model_save],steps_per_epoch = len(train_batches))#

## =====================================
## [2.5] Testando o modelo
print(f"evaluate: {model.evaluate(test_batches)}")