import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import sys 
import json 
import os
import config as cf
assert cf
import data_mp1.utils as ut 
import glob

np.set_printoptions(threshold=sys.maxsize)

def Mask2Detection(mask, img_id):

    label_image = label(mask, connectivity=1)
    predicciones = []
    
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

#PRUEBA
matrix_1, matrix_2, matrix_3= np.zeros((50,50)), np.zeros((50,50)), np.zeros((50,50))

matrix_1[3:9,2:10],matrix_1[2,3:9],matrix_1[9,3:9],matrix_1[1,4:8],matrix_1[10,4:8] =1, 1,1,1,1
matrix_1[17:33,15:35], matrix_1[14,17:32], matrix_1[35,17:32], matrix_1[15:17,16:34],matrix_1[33:35,16:34], matrix_1[13,19:30], matrix_1[36,19:30] =1, 1,1,1,1,1,1 
matrix_2[2:10,2:10], matrix_2[25:30,25:30]=1,1 #ES UN CUADRADO!!!!!! (CÍRCULO GRANDE)
matrix_3[0:5,12:30], matrix_3[25:30,25:30]=1,1 #ES UN CUADRADO !!!!!!!!
masks = [matrix_1, matrix_2, matrix_3]
lis_json = []
i= 0
fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(20, 12))
for mask in masks:
    ax[i,0].imshow(mask)
    ax[i,0].set_axis_off()
    ax[i,1].imshow(mask)
    ax[i,1].set_axis_off()
    predicciones = Mask2Detection(mask, i)
    
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

"""
FUNCIÓN 2
"""
var = "train"
carpeta =os.path.join("data_mp1","BCCD",var,"_annotations.coco.json")
predicciones = os.path.join("data_mp1","dummy_predictions.json")

train = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\train','*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\test','*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp1\\BCCD\\valid','*.jpg'))]

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
    
    TP, FP, FN, TN = 0, 0, 0, 0
    with open(annot_file) as f:
        annotations = json.load(f)
    with open(pred_file) as f:
        predictions = json.load(f)
    print("# de anotaciones", len(annotations["annotations"]))
    print("# de predicciones", len(predictions))
detections_Codigo1_Codigo2()

# Cálculo de las métricas
"""
precision = TP / TP_and_FP
cobertura = TP / TP_and_FN
f_medida = 2 * precision * cobertura / (precision + cobertura)

print("La precision para la clase de Jazz es: " + str(precision))
print("La cobertura para la clase de Jazz es: " + str(cobertura))
print("La F-medida para la clase de Jazz es: " + str(f_medida))
"""




    
