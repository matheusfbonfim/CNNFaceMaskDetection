## IMPORTANDO BIBLIOTECAS

import numpy as np
# Sequential model 
from keras.models import Sequential
# Bibliotecas - Camadas
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten, Dropout
# Otimizador
# from keras.optimizers import Adam
# Funcão de custo
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
# Matriz de confusão
from sklearn.metrics import confusion_matrix

import itertools
import os 
import shutil
import matplotlib.pyplot as plt

# ------------------------------------------------------------


#########################
## PRÉ-PROCESSAMENTO DE DADOS DE IMAGEM

# Dataset: https://www.kaggle.com/omkargurav/face-mask-dataset
#   Organizaremos nossos dados em conjuntos de treinamento, validação e teste -> Manualmente
#   Movendo subconjuntos de dados para subdiretórios para cada conjunto de dados separado.

# Caminhos para os dados de treinamento, validação e teste
path_train = '../database/Face Mask Dataset/Train'        # Caminho - Train
path_valid = '../database/Face Mask Dataset/Test'   # Caminho - Validação
path_test = '../database/Face Mask Dataset/Validation'          # Caminho - Teste


# -------------------------
### Gerando o conjunto de dados de treinamento, validação/teste

train_gen = ImageDataGenerator(rescale = 1.0/255.0)
valid_gen = ImageDataGenerator(rescale = 1.0/255.0)
test_gen =  ImageDataGenerator(rescale = 1.0/255.0)

train_batches = train_gen.flow_from_directory(directory = path_train, batch_size = 64 , color_mode= 'rgb' ,target_size=(200, 200), class_mode = 'binary')
valid_batches = valid_gen.flow_from_directory(directory = path_valid, batch_size = 64 , color_mode= 'rgb' ,target_size=(200, 200), class_mode = 'binary')
test_batches = test_gen.flow_from_directory(directory = path_test, batch_size = 64,  color_mode= 'rgb', target_size=(200, 200), class_mode = 'binary')


# Vendo os rotulos e os dados gerados
train_batches.class_indices # Rede neural precisa de indices categoricos em formato inteiro



