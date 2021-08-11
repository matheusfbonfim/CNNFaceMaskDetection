## IMPORTANDO BIBLIOTECAS

import numpy as np
# Sequential model 
#from keras.models import Sequential
# Bibliotecas - Camadas
#from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten, Dropout
# Otimizador
#from keras.optimizers import Adam
# Funcão de custo
#from keras.metrics import categorical_crossentropy
#from keras.preprocessing.image import ImageDataGenerator
# Matriz de confusão
#from sklearn.metrics import confusion_matrix

import itertools
import os 
import shutil
import matplotlib.pyplot as plt

# ------------------------------------------------------------


#########################
## PREPARANDO OS DADOS

# Dataset: https://www.kaggle.com/omkargurav/face-mask-dataset
#   Organizaremos nossos dados em conjuntos de treinamento, validação e teste -> Manualmente
#   Movendo subconjuntos de dados para subdiretórios para cada conjunto de dados separado.

# ----------
# Diretório
#   Localização dos diferentes conjuntos de dados



path = '../database/annotations/'





