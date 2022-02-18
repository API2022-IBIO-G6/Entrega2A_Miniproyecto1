# imports
from socket import TCP_NODELAY
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import config as cf
assert cf
import data_mp1.utils as ut
#import utils
import sys 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
import json
import os

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
# Inicializamos tres matrices 50x50 de ceros 
matrix_1, matrix_2, matrix_3= np.zeros((50,50)), np.zeros((50,50)), np.zeros((50,50))

#Imagen Binaria 1
#Dibujamos circulo 1
matrix_1[3:9,2:10],matrix_1[[2,9],3:9],matrix_1[[1,10],4:8] =1,1,1
#Dibujamos circulo 2
matrix_1[17:33,9:33], matrix_1[[16,33],11:31], matrix_1[[15,34],13:29] =1,1,1

# Imagen Binaria 2
matrix_2[[19,20,39,40], 22:38], matrix_2[[18,41], 24:36], matrix_2[[21,22,37,38], 21:39], matrix_2[23:37, 20:40] = 1,1,1,1

# Imagen Binaria 3
matrix_3[21:30, [14,15,34,35]], matrix_3[22:29, [13,36]], matrix_3[20:31, 16:34] = 1, 1, 1  # ES UN CUADRADO !!!!!!!!

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
plt.suptitle("Validación de la función de predicción de detección")
plt.show()

# 8.3 Función Evaluación de predicciones
var = "valid"
carpeta =os.path.join("data_mp1","BCCD",var,"_annotations.coco.json")
predicciones = os.path.join("data_mp1","dummy_predictions.json")

def get_iou(a, b, epsilon=1e-5):
    """ 
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner // tiene que representar w h 
        It returns the Intersect of Union score for these two boxes.
    Args: 
        a:          (list of 4 numbers) [x1,y1,x2,y2] // [x,y,w,h]
        b:          (list of 4 numbers) [x1,y1,x2,y2] // [x,y,w,h]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2])
    y2 = min(a[1]+a[3], b[1]+b[3])
    #x2 = min(a[2], b[2])
    #y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2]) * (a[3])
    area_b = (b[2]) * (b[3])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def detections_Codigo1_Codigo2(conf_thresh=0.5, jaccard_thresh=0.7, annot_file=carpeta, pred_file= predicciones):
    """
    @param conf_thresh: Umbral de confianza a partir del cual tener en cuenta una detección.
    @param jaccard_thresh: Umbral del índice de Jaccard a partir del cual se debe considerar una 
    detección como un verdadero positivo.
    @param annot_file: Ruta del archivo donde se encuentren las anotaciones.
    @param pred_file: Ruta del archivo donde se encuentren las predicciones.
    @return: TP: Cantidad de verdaderos positivos, FP: Cantidad de falsos positivos, FN: Cantidad de
    falsos negativos y TN: Cantidad de verdaderos negativos.
    """
    
    lista=[]
    with open(annot_file) as f:
        annotations = json.load(f)
    with open(pred_file) as f:
        predictions = json.load(f)
    
    TP, FP, FN, TN = 0, 0, 0, 0
    #Intersection Over Union (IOU)
    
    for pred in predictions:
        if pred["score"] >= conf_thresh:
            pred_bbox, pred_category, pred_image_id= pred["bbox"], pred["category_id"], pred["image_id"]
            for annot in annotations["annotations"]:
                annot_category, annot_image_id= annot["category_id"], annot["image_id"]
                if annot_image_id == pred_image_id:
                    annot_bbox = annot["bbox"]
                    iou = get_iou(pred_bbox, annot_bbox)
                    if iou >= jaccard_thresh:
                        if pred_category == annot_category:
                            TP += 1
                            lista.append("TP")
                    elif iou <= jaccard_thresh:
                        if pred_category == annot_category:
                            FP += 1 
                            lista.append("FP")         
        else:
            FN += 1 
            lista.append("FN")
    print("TP:",TP,"FP:",FP,"FN:",FN,"TN:",TN)
    print("\nlista:",lista, lista.count("TP"), lista.count("FP"), lista.count("FN"), lista.count("TN"))
    return TP,FP,FN, TN ## REVISAR FN!!!!!!!!!!

# 8.3.1 Validación de la función de evaluación de predicciones
"""
Utilizar los datos disponibles en el archivo “dummy_predictions.json” dentro de la carpeta data_mp1 
que les fue entregada. Con estos, y usando un umbral de confianza de 0.5 y un umbral del índice de 
Jaccard de 0.7, deben calcular las métricas especificadas en la función junto con las métricas de 
precisión, cobertura y f-medida asociadas. 
"""
TP,FP,FN, TN =detections_Codigo1_Codigo2()

precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)
f_measure = round(2*precision*recall/(precision+recall),2)
print("Precision:",precision,"Recall:",recall,"F-measure:",f_measure)

# HACER MATRIZ DE CONFUSIÓN ...

# 8.4 Función curva de precisión y recall
jacc_thr = [0.5,0.7,0.9]
sav_route = "data_mp1/curva_precision_recall.png"

def PRCurve_Codigo1_Codigo2(jaccard_thresh=jacc_thr[0], annot_file=carpeta, pred_file=predicciones, save_route=sav_route):
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
