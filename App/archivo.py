# imports
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import utils
import sys 
import  sklearn 
import json

# 8.2. Función predicciones de detección 

from ast import Param


def Mask2Detection_Codigo1_Codigo2(mask, img_id):
    """
    @param mask: Mascara binaria de la cual se extraerán las predicciones de anotación.
    @param img_id: Id de la imagen
    @return: Lista de diccionarios con las predicciones. Esta debe tener el mismo formato
    ejemplificado en el archivo “dummy_predictions.json”.
    """
    label_image = label(mask, connectivity=1)
    predicciones = []
    print("# de anotaciones detectadas", len(regionprops(label_image)))
    for region in regionprops(label_image):

        min_row, min_col, max_row, max_col = region.bbox
        
        print("print:",min_row, min_col, max_row, max_col)
        
        dic = {"image_id": img_id, 
            "category_id": 1,
            "bbox": [
                min_row,
                min_col,
                max_row-min_row,
                max_col -min_col
            ],
            "score": 1}
        predicciones.append(dic)
    
    return predicciones

# 8.2.1 Validación de la función de predicción de detección    
"""
Construir 3 imágenes binarias de 50x50 que en su interior tengan uno o más círculos por 
imagen (Cada circulo debe ser de diferente tamaño y estar en diferente posición). Usando 
las 3 imágenes binarias diseñadas, elaboren un subplot de 3x2 donde se pueda visualizar 
la máscara de segmentación original en la primera columna y la anotación de detección 
correspondiente en la otra columna.
"""
matrix_1, matrix_2, matrix_3= np.zeros((50,50)), np.zeros((50,50)), np.zeros((50,50))

matrix_1[2:10,2:10], matrix_1[25:30,25:30], matrix_1[35:40,40:46] =1, 1,1 
matrix_2[2:10,2:10], matrix_2[25:30,25:30]=1,1
matrix_3[0:5,12:30], matrix_3[25:30,25:30]=1,1
masks = [matrix_1, matrix_2, matrix_3]
lis_json = []
i= 0
fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(20, 12))
for mask in masks:
    ax[i,0].imshow(mask)
    ax[i,0].set_axis_off()
    ax[i,1].imshow(mask)
    ax[i,1].set_axis_off()
    predicciones = Mask2Detection_Codigo1_Codigo2(mask, i)
    
    for p in predicciones:
        print(p)
        rect = mpatches.Rectangle(((p["bbox"][1])-1.1, (p["bbox"][0])-1.1), (p["bbox"][3])+1.1, (p["bbox"][2])+1.1,
                                fill=False, edgecolor='red', linewidth=2)
        ax[i,1].add_patch(rect)
    i += 1
    lis_json += predicciones
    print("----------------\n",lis_json)
plt.show()
### OTRA OPCIÓN
i= 0
fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(20, 12))
for m in range(0,len(masks)):
    ax[i,0].imshow(masks[m])
    ax[i,0].set_axis_off()

    label_image = label(masks[m], connectivity=1)
    ax[i,1].imshow(label_image)
    ax[i,1].set_axis_off()

    for region in regionprops(label_image):
        
        min_row, min_col, max_row, max_col = region.bbox
        
        rect = mpatches.Rectangle((min_col-1, min_row-1), max_col - min_col+1, max_row - min_row+1,
                                fill=False, edgecolor='red', linewidth=3)
        ax[i,1].add_patch(rect)
    i += 1
plt.show()


# 8.3 Función Evaluación de predicciones
def detections_Codigo1_Codigo2(conf_thresh, jaccard_thresh, annot_file, pred_file):
    """
    @param conf_thresh: Umbral de confianza a partir del cual tener en cuenta una detección.
    @param jaccard_thresh: Umbral del índice de Jaccard a partir del cual se debe considerar una 
    detección como un verdadero positivo.
    @param annot_file: Ruta del archivo donde se encuentren las anotaciones.
    @param pred_file: Ruta del archivo donde se encuentren las predicciones.
    @return: TP: Cantidad de verdaderos positivos, FP: Cantidad de falsos positivos, FN: Cantidad de
    falsos negativos y TN: Cantidad de verdaderos negativos. 
    """
    with open(annot_file) as f:
        annotations = json.load(f)
    with open(pred_file) as f:
        predictions = json.load(f)
    
    TP, FP, FN, TN = 0, 0, 0, 0
    pass

# 8.3.1 Validación de la función de evaluación de predicciones
"""
Utilizar los datos disponibles en el archivo “dummy_predictions.json” dentro de la carpeta data_mp1 
que les fue entregada. Con estos, y usando un umbral de confianza de 0.5 y un umbral del índice de 
Jaccard de 0.7, deben calcular las métricas especificadas en la función junto con las métricas de 
precisión, cobertura y f-medida asociadas. 
"""


# 8.4 Función curva de precisión y recall
def PRCurve_Codigo1_Codigo2(jaccard_thresh, annot_file, pred_file, save_route):
    """
    @param jaccard_thresh: Umbral del indice de Jaccard a partir del cual se debe considerar una
    deteccion como un verdadero positivo.
    @param annot_file: Ruta del archivo donde se encuentren las anotaciones.
    @param pred_file: Ruta del archivo donde se encuentren las predicciones.
    @param save_route: Ruta donde se vaya a guardar la imagen.
    @return area_under_the_curve: El área por debajo de la curva de precisión y cobertura.
    @return data: Lista con precisiones y coberturas de todos los puntos de la curva.
    """
    pass

# 8.4.1 Validación de la función de curva de precisión y recall
"""
Haciendo uso de todas las funciones implementadas deberán evaluar esta última usando los datos
que tienen disponibles en la carpeta data_mp1 en el archivo “dummy_predictions.json”. Usando estos
datos realicen 3 curvas de precisión y cobertura e incluyanlas en una misma Figura. Para esto utilicen
3 índices de Jaccard diferentes [0.5, 0.7, 0.9]. 
"""
