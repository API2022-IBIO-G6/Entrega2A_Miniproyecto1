#import cv as cv
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import utils
import sys 

np.set_printoptions(threshold=sys.maxsize)

def Mask2Detection(mask, img_id):

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

#PRUEBA
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

    
    






    
