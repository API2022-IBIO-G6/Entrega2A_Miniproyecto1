# imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import json
import os
from scipy.integrate import simps

# 8.2. Función predicciones de detección 

def Mask2Detection_201923972_201923531(mask, img_id):
    """
    @param mask: Mascara binaria de la cual se extraerán las predicciones de anotación.
    @param img_id: Id de la imagen
    @return: Lista de diccionarios con las predicciones. Esta debe tener el mismo formato
    ejemplificado en el archivo “dummy_predictions.json”.
    """
    predicciones = []
    # Obtenemos una matriz etiquetada donde a todas las regiones conectadas se les asigna el mismo valor entero
    label_image = label(mask, connectivity=1) 
    print("# de anotaciones detectadas", len(regionprops(label_image)))

    # Recorremos cada una de las regiones conectadas 
    for region in regionprops(label_image):

        #Obtenemos las coordenadas del rectangulo dilimitador de la región []
        min_row, min_col, max_row, max_col = region.bbox
        
        print("El rectangulo delimitador está en el intervalo [{}, {}), [{},{})".format(min_row, max_row, min_col, max_col))
        
        # Creamos el diccionario asociado a la predicción a una prediccion
        dic = {"image_id": img_id, 
            "category_id": 3,
            "bbox": [
                min_row,
                min_col,
                max_row-min_row, # ancho
                max_col-min_col  # alto
            ],
            "score": 1 #valor arbitrario 
            } 
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
print( "8.2.1 Validación de la función de predicción de detección " )
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
matrix_3[21:30, [14,15,34,35]], matrix_3[22:29, [13,36]], matrix_3[20:31, 16:34] = 1, 1, 1  

masks = [matrix_1, matrix_2, matrix_3]
lis_json = []
i= 0 # variable para el img_id
fig, ax = plt.subplots(nrows=3,ncols=2)
for mask in masks:
    # Se grafica la amscara de segmentación original
    ax[i,0].imshow(mask)
    ax[i,0].set_axis_off()
    ax[i,1].imshow(mask)
    ax[i,1].set_axis_off()
    # se obtiene la prediccion asociada
    print("\n*******Imagen Binaria {}*********\n".format(i+1))
    predicciones = Mask2Detection_201923972_201923531(mask, i)    
    for p in predicciones:
        # Se grafica la anotación correspondiente
        rect = mpatches.Rectangle(((p["bbox"][1])-1.1, (p["bbox"][0])-1.1), (p["bbox"][3])+1.1, (p["bbox"][2])+1.1,
                                fill=False, edgecolor='red', linewidth=2)
        ax[i,1].add_patch(rect)
    i += 1
    lis_json += predicciones
plt.suptitle("Validación de la función de predicción de detección")
plt.tight_layout()
plt.show()

# 8.3 Función Evaluación de predicciones
var = "valid"
anotaciones =os.path.join("data_mp1","BCCD",var,"_annotations.coco.json")
predicciones = os.path.join("data_mp1","dummy_predictions.json")

def get_iou(a, b, epsilon=1e-5):
    """ 
    Dados dos bbox `a` y `b` definidas cada una como una lista de cuatro valores de coordenadas:
            [x1,y1,x2,y2]
        donde:
            x1,y1 representan las coordenadas de la esquina superior izquierda
            x2,y2 representan el ancho (w) y el alto (h) 
        epsilon:    (float) pequeño valor para prevenir división por cero
    Retorno:
        (float)  retorna el score para la intercecion de la union para las dos bbox 
    """
    # COORDENADAS DEL RECTANGULO DE INTERCECION
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2])
    y2 = min(a[1]+a[3], b[1]+b[3])

    # AREA DE SOBRELAPAMIENTO- Area donde las dos box se intersectan
    width = (x2 - x1)
    height = (y2 - y1)
    # Si no hay sobrelapamiento retornamos 0
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # AREA DE LA UNION
    area_a = (a[2]) * (a[3])
    area_b = (b[2]) * (b[3])
    area_combined = area_a + area_b - area_overlap

    # Proporción Area de la intersecion / Area de la unión
    iou = area_overlap / (area_combined+epsilon)
    return iou


def detections_201923972_201923531(conf_thresh=0.5, jaccard_thresh=0.7, annot_file=anotaciones, pred_file= predicciones):
    """
    @param conf_thresh: Umbral de confianza a partir del cual tener en cuenta una detección.
    @param jaccard_thresh: Umbral del índice de Jaccard a partir del cual se debe considerar una 
    detección como un verdadero positivo.
    @param annot_file: Ruta del archivo donde se encuentren las anotaciones.
    @param pred_file: Ruta del archivo donde se encuentren las predicciones.
    @return: TP: Cantidad de verdaderos positivos, FP: Cantidad de falsos positivos, FN: Cantidad de
    falsos negativos y TN: Cantidad de verdaderos negativos.
    """
    
    with open(annot_file) as f: # abrimos el archivo de anotaciones
        annotations = json.load(f) # Se guarda en una variable anotaciones 
    with open(pred_file) as f: # Se abre el archivo de predicciones
        predictions = json.load(f) # Se guarda el archivo de predicciones en la variable predictions
    
    TP, FP, FN = 0, 0, 0
    for pred in predictions:
        # Se recorre el archivo de predicciones y se obtiene el id de la imagen, su categoría y su bbox    
        pred_bbox, pred_category, pred_image_id= pred["bbox"], pred["category_id"], pred["image_id"]
        # Se recorre el archivo de anotaciones 
        for annot in annotations["annotations"]:
            annot_category, annot_image_id= annot["category_id"], annot["image_id"]
            # Se verifica que la imagen de la predicción sea la misma que la de la anotación
            # y que la categoría de la predicción sea la misma que la de la anotación (en este caso 3)
            if annot_image_id == pred_image_id:
                if pred_category == annot_category:
                    annot_bbox = annot["bbox"]
                    #Se obtiene la intersección sobre la unión con la función iou 
                    iou = get_iou(pred_bbox, annot_bbox)
                    # Si el score de la predicción es mayor al umbral de confianza se cuenta la imagen 
                    if pred["score"] >= conf_thresh:
                        # Si el iou es mayor al umbral de jaccard se cuenta como un TP
                        if iou >= jaccard_thresh:
                            TP += 1
                        # Si el iou es menor al umbral de jaccard se cuenta como un FP
                        elif iou <= jaccard_thresh:
                            FP += 1        
                    # Si no cumple con lo anterior entonces la anotación no fue detectada y se cuenta como FN 
                    else:
                        FN += 1
    TN = 0   # No se tiene en cuenta TN para detección de objetos                 
    return TP,FP,FN, TN 

# 8.3.1 Validación de la función de evaluación de predicciones
"""
Utilizar los datos disponibles en el archivo “dummy_predictions.json” dentro de la carpeta data_mp1 
que les fue entregada. Con estos, y usando un umbral de confianza de 0.5 y un umbral del índice de 
Jaccard de 0.7, deben calcular las métricas especificadas en la función junto con las métricas de 
precisión, cobertura y f-medida asociadas. 
"""
TP,FP,FN,TN =detections_201923972_201923531()
print("\n8.3.1 Validación de la función de evaluación de predicciones")
print("\nMatriz de Confusión: Anotacion (A), Prediccion (P)")
matrix = np.array([[TP,FP],[FN,TN]])
print("{:25}\t{:20}\t{:20}\t".format("", "(A) Celulas blancas","(A) NO Celulas blancas" ))
print("{:15}\t{:20}\t{:20}\t".format("(P) Celulas blancas", TP, FP))
print("{:15}\t{:20}\t{:20}\t".format("(P) NO Celulas blancas", FN, TN))

print("\nMatriz de Confusión Normalizada por filas: Anotacion (A), Prediccion (P)")
row_sums = matrix.sum(axis=1) #sumamos las filas
normalize_row = np.round(matrix / row_sums[:, np.newaxis], 2) #dividimos la matriz entre la suma de fila
print("{:25}\t{:20}\t{:20}\t".format("", "(A) Celulas blancas","(A) NO Celulas blancas" ))
print("{:15}\t{:20}\t{:20}\t".format("(P) Celulas blancas", normalize_row[0, 0], normalize_row[0, 1]))
print("{:15}\t{:20}\t{:20}\t".format("(P) NO Celulas blancas", normalize_row[1, 0], normalize_row[1, 1]))

print("\nMatriz de Confusión Normalizada por columnas: Anotacion (A), Prediccion (P)")
col_sums = matrix.sum(axis=0) #sumamos las columnas
normalize_col = np.round(matrix / col_sums, 2) #dividimos la matriz entre la suma de columnas
print("{:25}\t{:20}\t{:20}\t".format("", "(A) Celulas blancas","(A) NO Celulas blancas" ))
print("{:15}\t{:20}\t{:20}\t".format("(P) Celulas blancas", normalize_col[0, 0], normalize_col[0, 1]))
print("{:15}\t{:20}\t{:20}\t".format("(P) NO Celulas blancas", normalize_col[1, 0], normalize_col[1, 1]))

#Métricas de precisión, cobertura y f-medida asociadas.
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)
f_measure = round(2*precision*recall/(precision+recall),2)

print("\nPara un Umbral de confianza de 0.5 y un Umbral del índice de Jaccard de 0.7")
print("Precisión:",precision,"Cobertura:",recall,"F-medida:",f_measure)

# 8.4 Función curva de precisión y recall
jacc_thr = [0.5,0.7,0.9]
sav_route = "data_mp1/curva_precision_recall.png"

def PRCurve_201923972_201923531(jaccard_thresh=jacc_thr, annot_file=anotaciones, pred_file=predicciones, save_route=sav_route):
    """
    @param jaccard_thresh: Umbral del indice de Jaccard a partir del cual se debe considerar una
    deteccion como un verdadero positivo.
    @param annot_file: Ruta del archivo donde se encuentren las anotaciones.
    @param pred_file: Ruta del archivo donde se encuentren las predicciones.
    @param save_route: Ruta donde se vaya a guardar la imagen.
    @return area_under_the_curve: El área por debajo de la curva de precisión y cobertura.
    @return data: Lista con precisiones y coberturas de todos los puntos de la curva.
    """
    data =[]
    # Se crea una lista llamada data y se comienza a reccorerel umbrales de jaccard
    for j in jaccard_thresh:
        precision = []
        recall = [] # Se crean dos listas. Una de precisión y otra de cobertura
        for i in np.arange(0.0, 1.1, 0.01):
            TP, FP, FN, TN = detections_201923972_201923531(conf_thresh=i, jaccard_thresh=j)
            # Se recorrren los umbrales de confianza y se calculan las métricas de precisión y cobertura
            if TP+FP != 0: # No se tiene en cuenta división por 0.0
                precision.append(TP/(TP+FP))
                recall.append(TP/(TP+FN))
        area_under_the_curve = simps(precision,dx=0.01) # Se calcula el área bajo la curva con simps
        
        print("Área debajo de la curva para el índice de Jaccard de {}: {}".format(j, round(area_under_the_curve,2)))
        
        data.append({"precision":precision,"cobertura":recall, "area_under_the_curve":area_under_the_curve})
        # Se crea una lista con los datos de precision y cobertura de cada punto de la curva y el área bajo la curva
        plt.plot(recall, precision,label="Jaccard = {}".format(j)) # Se grafica la curva
    
    # Se grafica la curva de precisión y cobertura para los 3 índices de Jaccard en un mismo plot 
    plt.xlabel("Cobertura")
    plt.ylabel("Precisión")
    plt.title("Curva precisión-cobertura")
    plt.legend()
    plt.savefig(save_route) # se guarda la grafica en la ruta especificada
    plt.grid()
    plt.show()
    return data # Se retorna una lista de diccionarios con la información de cada punto de la curva

# 8.4.1 Validación de la función de curva de precisión y recall
"""
Haciendo uso de todas las funciones implementadas deberán evaluar esta última usando los datos
que tienen disponibles en la carpeta data_mp1 en el archivo “dummy_predictions.json”. Usando estos
datos realicen 3 curvas de precisión y cobertura e incluyanlas en una misma Figura. Para esto utilicen
3 índices de Jaccard diferentes [0.5, 0.7, 0.9]. 
"""
print("\n8.4.1 Validación de la función de curva de precisión y recall\n")
PRCurve_201923972_201923531() 
