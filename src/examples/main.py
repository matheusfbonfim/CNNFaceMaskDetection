"""
Display the webcam in Python using OpenCV (cv2)
Face Mask Detection
"""

## =====================================
## [0] IMPORTANDO BIBLIOTECAS

# OpenCV - biblioteca de funções de programação voltada
#          principalmente para a visão computacional em tempo real.
import cv2

# Bibliotecas - Manipulação de dados
import numpy as np

# Adiciona suporte para abrir, manipular e salvar muitos formatos de arquivo de imagem diferentes.
#import PIL import Image

# Reconhecimento da face de uma pessoa - Rede Neural
from mtcnn.mtcnn import MTCNN

# Carregar o modelo
from keras.models import load_model

import os

## =====================================
## =====================================
## [1] DEFINIÇOES INICIAIS

# Carregando o modelo
model = load_model("../model/new-face-mask-detection.h5")

# Forma de entrada na qual treinamos nosso modelo
input_shape = (200,200,3)

# Labels de saida
labels_output = {0: "WithMask", 1: "WithoutMask"}

# Cores para indicação
#               0  -   RED       1 -   GREEN
labels_colors = {0: (0, 255, 0), 1: (0,0,255)}

# Detector de faces
detector = MTCNN()

# Tamanho
size = 4

# Capturando imagem da camera (0)
cam = cv2.VideoCapture(0)

## =====================================
## =====================================
## [2] FUNÇÃO PARA DETECÇÃO

def webcam_detector():
    # Lendo frame a frame
    while True:
        (ret_val, img) = cam.read()

        # print(f"Ret_val: {ret_val}")
        # print(f"Imagem: {img}")

        # Redimensionando a imagem para acelerar a detecção
        image_reduce = cv2.resize(img, (img.shape[1] // size, img.shape[0] // size))

        # O MTCNN precisa do arquivo no formato RGB, mas o cv2 lê no formato BGR. Portanto, estamos convertendo
        rgb_image = cv2.cvtColor(image_reduce, cv2.COLOR_BGR2RGB)

        # Detectando faces - teremos coordenadas (x, y, w, h)
        n_faces = detector.detect_faces(image_reduce)

        #print(f"Faces: {n_faces}") # Imprime a localizacao da face em 'box'

        # Desenhe retângulos ao redor de cada rosto
        for f in n_faces:
            x, y, w, h = [v * size for v in f['box']] # Em cada elemento do box multiplicar por 4

            # Cortar a parte do rosto de toda a imagem
            face_image = img[y: y + h, x: x + w]

            # print(f"FaceImage: {face_image}")

            # Redimensionar a imagem para nosso tamanho de entrada revisto no qual treinamos nosso modelo
            redimensiona = cv2.resize(face_image, (input_shape[0], input_shape[1]))

            # Usamos ImageDataGenerator e treinamos nosso modelo em lotes
            # Portanto, a forma de entrada para nosso modelo é (batch_size, height, width, color_depth)
            # Estamos convertendo a imagem para este formato, ou seja (altura, largura, profundidade_de_ cor) -> (batch_size, height, width, color_depth)
            reshaped = np.reshape(redimensiona, (1, input_shape[0], input_shape[1], 3))

            # Predicao
            predict = model.predict(reshaped)
            print(f"Predicao: {predict}")

            # Obtendo o índice para o valor máximo
            predict_label = np.argmax(predict, axis = 1)[0]

            # Caixa delimitadora (grande retângulo ao redor da face)
            cv2.rectangle(img,(x, y), (x + w, y +h), labels_colors[predict_label], 2)

            # Retângulo pequeno acima do BBox, onde colocaremos nosso texto
            cv2.rectangle(img, (x, y - 40), (x + w, y + h), labels_colors[predict_label], -1)

            # Espessura de -1 px irá preencher o retângulo com a cor especificada

            cv2.putText(img, labels_output[predict_label],(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Mostre a imagem
        cv2.imshow("Detector - Mascara", img)
        key = cv2.waitKey(10)

        # Se pressionar o ESC sai do loop
        if key == 27:
            break

    # # Stop video
    cam.release()
    cv2.destroyAllWindows()


def main():
    webcam_detector()


if __name__ == '__main__':
    main()